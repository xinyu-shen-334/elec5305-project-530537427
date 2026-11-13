"""Speaker-disjoint development/test comparison of STFT Wiener gain floors.
Run: python src/experiment.py [--stage development|test|all]
Sources and selected clips are shipped with this project; no training is required.
"""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import argparse,csv,json,re,time,hashlib,platform
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft,istft
from pystoi import stoi
from faster_whisper import WhisperModel
FS=16000;FFT=512;HOP=128;EPS=1e-12
MODEL_REVISION='0d3d19a32d3338f10357c0889762bd8d64bbdeba'
ROOT=Path(__file__).resolve().parents[1]
NOISES=['white','traffic','cafeteria'];LEVELS=[-5,0,5,10];FLOORS=[.05,.10,.20,.35]
def spec(x):return stft(x,fs=FS,window='hann',nperseg=FFT,noverlap=FFT-HOP,boundary='zeros',padded=True)[2]
def psd(x):return np.mean(abs(spec(x))**2,axis=1)
def enhance(y,p,floor):
 Y=spec(y);P=p[:,None];S=np.maximum(abs(Y)**2-P,0)
 G=np.maximum(floor,S/(S+P+EPS))
 x=istft(G*Y,fs=FS,window='hann',nperseg=FFT,noverlap=FFT-HOP,boundary=True)[1]
 return x[:len(y)],G

def snr(s,x):return float(10*np.log10((np.mean(s*s)+EPS)/(np.mean((s-x)**2)+EPS)))
def words(t):return re.findall(r'[a-z0-9]+',t.lower().replace("'",''))
def word_errors(reference,hypothesis):
 a,b=words(reference),words(hypothesis);d=np.zeros((len(a)+1,len(b)+1),dtype=int)
 d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(x!=y))
 i,j=len(a),len(b);S=D=I=0
 while i or j:
  if i and j and a[i-1]==b[j-1] and d[i,j]==d[i-1,j-1]:i-=1;j-=1
  elif i and j and d[i,j]==d[i-1,j-1]+1:S+=1;i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:D+=1;i-=1
  else:I+=1;j-=1
 return dict(errors=S+D+I,substitutions=S,deletions=D,insertions=I,nwords=len(a))

def load_sources():
 manifest=json.loads((ROOT/'data/manifest.json').read_text());noises={}
 for key,name in [('traffic','STRAFFIC'),('cafeteria','PCAFETER')]:
  x,fs=sf.read(ROOT/'data/noise'/(name+'.wav'));assert fs==FS and x.ndim==1
  noises[key]=x
 return manifest,noises

def mixture(record,kind,level,noises,repeat=0):
 s,fs=sf.read(ROOT/record['file']);assert fs==FS
 # Stable seeds: independent of enumeration order, Python hash and chosen floor.
 key=f"5305|{record['split']}|{record['id']}|{kind}|{level}|{repeat}"
 seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
 n=len(s)
 if kind=='white':
  full=rng.normal(size=n+FS);offset=-1
 else:
  # Development 0-60s; test 120-300s, leaving a 60s guard interval.
  low,high=(0,60*FS) if record['split']=='development' else (120*FS,len(noises[kind]))
  offset=int(rng.integers(low,high-n-FS+1));full=noises[kind][offset:offset+n+FS].copy()
 full-=np.mean(full);cal=full[:FS];v=full[FS:]
 alpha=np.sqrt(np.mean(s*s)/(np.mean(v*v)*10**(level/10)))
 cal=alpha*cal;v=alpha*v;y=s+v
 return s,y,v,cal,{'seed':seed,'noise_start_sample':offset,'noise_alpha':float(alpha)}

def write_csv(rows,path):
 if rows:
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','development','test'],default='all');args=parser.parse_args()
 out=ROOT/'results';out.mkdir(exist_ok=True)
 manifest,noises=load_sources()
 assert not ({r['speaker_id'] for r in manifest if r['split']=='development'} & {r['speaker_id'] for r in manifest if r['split']=='test'})
 print('Loading fixed Whisper tiny.en recogniser',flush=True)
 model=WhisperModel('tiny.en',revision=MODEL_REVISION,device='cpu',compute_type='int8',cpu_threads=4,num_workers=1)
 decoding=dict(language='en',beam_size=1,temperature=0,condition_on_previous_text=False,vad_filter=False,no_repeat_ngram_size=3,max_new_tokens=96)
 def transcribe(x):
  segments,_=model.transcribe(x.astype(np.float32),**decoding)
  return ' '.join(seg.text for seg in segments).strip()
 started=time.perf_counter()
 for stage in ['development','test']:
  if args.stage not in ['all',stage]:continue
  if stage=='development':
   methods=[('noisy',None)]+[(f'floor{round(f*100):03d}',f) for f in FLOORS]
  else:
   choice=json.loads((out/'development_selection.json').read_text());selected=choice['selected_floor']
   methods=[('noisy',None),('baseline',.05),('selected',selected),('oracle',.05)]
  log=out/(stage+'_records.jsonl');rows=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
  done={(r['id'],r['noise'],r['input_snr'],r['method']) for r in rows}
  subset=[r for r in manifest if r['split']==stage]
  def record_output(rec,s,x,kind,level,method,params,seconds):
   text=transcribe(x);metrics=word_errors(rec['text'],text)
   row=dict(id=rec['id'],speaker_id=rec['speaker_id'],sex=rec['sex'],split=stage,noise=kind,input_snr=level,method=method,snr=snr(s,x),stoi=1. if kind=='clean' else float(stoi(s,x,FS,extended=False)),**metrics,transcript=text,reference=rec['text'],enhancement_seconds=seconds,**params)
   rows.append(row)
   with log.open('a') as f:f.write(json.dumps(row)+'\n')
   done.add((rec['id'],kind,level,method))
  for ui,rec in enumerate(subset):
   s,fs=sf.read(ROOT/rec['file'])
   if (rec['id'],'clean',99,'clean') not in done:record_output(rec,s,s,'clean',99,'clean',dict(seed=0,noise_start_sample=-1,noise_alpha=0.),0.)
   for kind in NOISES:
    for level in LEVELS:
     s,y,v,cal,params=mixture(rec,kind,level,noises);p=psd(cal);oracle=psd(v)
     outputs={};times={}
     for method,floor in methods:
      t0=time.perf_counter();outputs[method]=y.copy() if floor is None else enhance(y,oracle if method=='oracle' else p,floor)[0];times[method]=time.perf_counter()-t0
     # All compared outputs and reference share one mixture-dependent scale.
     # Include every output to avoid accidental clipping from overlap-add peaks.
     scale=max(1.,float(np.max(abs(s))),*(float(np.max(abs(x))) for x in outputs.values()))
     for method,x in outputs.items():
      if (rec['id'],kind,level,method) not in done:record_output(rec,s/scale,x/scale,kind,level,method,params,times[method])
   write_csv(rows,out/(stage+'_measurements.csv'))
   print(stage,ui+1,'/',len(subset),'completed; elapsed',round(time.perf_counter()-started),'s',flush=True)
  if stage=='development':
   scores={}
   for method,floor in methods:
    if floor is None:continue
    rs=[r for r in rows if r['method']==method]
    scores[str(floor)]={'wer':100*sum(r['errors'] for r in rs)/sum(r['nwords'] for r in rs),'errors':sum(r['errors'] for r in rs),'nwords':sum(r['nwords'] for r in rs)}
   # Select lowest dev corpus WER; ties prefer the smaller, more suppressive floor.
   selected=min(FLOORS,key=lambda f:(scores[str(f)]['wer'],f))
   (out/'development_selection.json').write_text(json.dumps({'scores':scores,'selected_floor':selected,'criterion':'minimum development corpus WER, ties select smaller floor','test_data_used':False},indent=2))
   print('Frozen development selection:',selected,flush=True)
 import importlib.metadata as md
 metadata={'python':platform.python_version(),'platform':platform.platform(),'model':'Systran/faster-whisper-tiny.en','model_revision':MODEL_REVISION,'compute_type':'int8','cpu_threads':4,'decoding':decoding,'versions':{k:md.version(k) for k in ['numpy','scipy','soundfile','pystoi','faster-whisper','ctranslate2']},'elapsed_seconds':time.perf_counter()-started,'selection_and_data':'data/manifest.json; data/sources.json','stage':args.stage}
 (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
