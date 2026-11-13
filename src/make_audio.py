"""Export fair, jointly scaled audio comparisons and acoustic figures."""
import json,csv,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from experiment import load_sources,mixture,psd,enhance,spec,FS,ROOT

def main():
 manifest,noises=load_sources();choice=json.loads((ROOT/'results/development_selection.json').read_text());floor=choice['selected_floor'];A=ROOT/'audio';A.mkdir(exist_ok=True)
 test=[r for r in manifest if r['split']=='test'];examples=[]
 for sex in ['F','M']:
  rec=next(r for r in test if r['sex']==sex)
  for kind in ['traffic','cafeteria']:examples.append((rec,kind,0))
 audio_manifest=[]
 for ei,(rec,kind,level) in enumerate(examples,1):
  s,y,v,cal,params=mixture(rec,kind,level,noises);p=psd(cal)
  b,gb=enhance(y,p,.05);z,gz=enhance(y,p,floor);o,go=enhance(y,psd(v),.05)
  waves={'clean':s,'noisy':y,'baseline':b,'selected':z,'oracle':o}
  # Common gain for ALL versions of this clip; never independently normalise.
  common=.9/max(float(np.max(abs(x))) for x in waves.values());waves={k:x*common for k,x in waves.items()}
  dest=A/f"example_{ei:02d}_{rec['sex']}_{kind}_0dB";dest.mkdir(exist_ok=True)
  for name,x in waves.items():sf.write(dest/(name+'.wav'),x,FS,subtype='PCM_24')
  silence=np.zeros(FS);sequence=np.concatenate([item for name in ['clean','noisy','baseline','selected'] for item in [waves[name],silence]])
  sf.write(dest/'comparison.wav',sequence,FS,subtype='PCM_24')
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(dest/'comparison.wav'),'-codec:a','libmp3lame','-b:a','128k',str(dest/'comparison.mp3')],check=True)
  audio_manifest.append(dict(example=ei,id=rec['id'],sex=rec['sex'],noise=kind,input_snr=level,transcript=rec['text'],selected_floor=floor,common_export_gain=common,folder=dest.relative_to(ROOT).as_posix(),sequence=['clean','noisy','baseline','selected'],separator_seconds=1))
  if ei==1:
   # One full-size example, selected before looking at performance.
   fig,axes=plt.subplots(2,2,figsize=(10.5,4.5),sharex=True,sharey=True)
   for ax,name,label in zip(axes.flat,['clean','noisy','baseline','selected'],['Clean reference','Noisy input','Baseline floor 0.05',f'Development-selected floor {floor:.2f}']):
    Z=spec(waves[name]);t=np.arange(Z.shape[1])*128/FS;f=np.arange(Z.shape[0])*FS/512
    mesh=ax.pcolormesh(t,f,20*np.log10(np.maximum(abs(Z),1e-7)),vmin=-90,vmax=-15,cmap='magma',shading='auto');ax.set(title=label,ylim=(0,6000),ylabel='Frequency (Hz)',xlabel='Time (s)')
   fig.tight_layout();fig.savefig(ROOT/'results/audio_spectrograms.png',dpi=190);plt.close(fig)
   fig,axes=plt.subplots(1,2,figsize=(10.5,3),sharey=True)
   for ax,G,label in [(axes[0],gb,'Baseline'),(axes[1],gz,'Selected floor')]:
    im=ax.imshow(G,origin='lower',aspect='auto',vmin=0,vmax=1,extent=[0,len(s)/FS,0,FS/2],cmap='viridis');ax.set(title=label,xlabel='Time (s)',ylim=(0,6000))
   axes[0].set_ylabel('Frequency (Hz)');fig.colorbar(im,ax=axes.ravel().tolist(),label='Amplitude gain',pad=.02);fig.savefig(ROOT/'results/gain_comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)
   times=np.arange(len(s))/FS;fig,axes=plt.subplots(4,1,figsize=(10.5,5),sharex=True,sharey=True)
   for ax,name in zip(axes,['clean','noisy','baseline','selected']):ax.plot(times,waves[name],lw=.35,color='#137f88');ax.set(ylabel=name)
   axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(ROOT/'results/audio_waveforms.png',dpi=180);plt.close(fig)
 # Plot noise spectra at equal RMS to avoid interpreting source-level differences.
 fig,ax=plt.subplots(figsize=(10.5,2.8));rng=np.random.default_rng(5305)
 for name,x in [('white',rng.normal(size=10*FS)),('traffic',noises['traffic'][:10*FS]),('cafeteria',noises['cafeteria'][:10*FS])]:
  x=x-np.mean(x);x/=np.sqrt(np.mean(x*x));f,p=welch(x,FS,nperseg=1024);ax.plot(f,10*np.log10(np.maximum(p,1e-15)),label=name)
 ax.set(xlim=(0,8000),xlabel='Frequency (Hz)',ylabel='PSD (dB/Hz; unit RMS input)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/noise_spectra.png',dpi=190);plt.close(fig)
 (A/'audio_manifest.json').write_text(json.dumps(audio_manifest,indent=2))
 text='AUDIO LISTENING GUIDE\n\nExamples were selected by first test ID within sex, not by achieved improvement.\nEach example has traffic or cafeteria noise at 0 dB.\ncomparison.wav / comparison.mp3 play CLEAN -> NOISY -> BASELINE -> SELECTED,\nwith one second silence after each. All versions share one export gain.\nDo not compare independently normalised volumes: there is no such normalisation here.\nThe oracle output is a diagnostic and is excluded from the main listening sequence.\nThe selected output is tuned for development WER, not listener preference.\nNoise reduction may improve SNR without improving every spoken word.\n\nData: LibriSpeech CC BY 4.0; DEMAND CC BY-SA 3.0.\nAudio mixtures and derived enhanced examples are supplied under CC BY-SA 3.0,\nwith both source attributions and the processing changes described in the report.\n'
 (A/'LISTENING_GUIDE.txt').write_text(text)
 print('Exported',len(examples),'five-way audio sets with comparison samplers')
if __name__=='__main__':main()
