"""Summarise paired test outputs with speaker-cluster bootstrap intervals."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
METHODS=['noisy','baseline','selected','oracle']
LABELS={'noisy':'Noisy','baseline':'Baseline floor 0.05','selected':'Development-selected floor','oracle':'Oracle noise PSD'}
COLORS={'noisy':'#5d6770','baseline':'#d86a3a','selected':'#137f88','oracle':'#7462a1'}
def load():
 rows=list(csv.DictReader((OUT/'test_measurements.csv').open()))
 for r in rows:
  for k in ['input_snr','snr','stoi','enhancement_seconds','noise_alpha']:r[k]=float(r[k])
  for k in ['speaker_id','seed','noise_start_sample','errors','substitutions','deletions','insertions','nwords']:r[k]=int(r[k])
 return rows

def aggregate(rs):
 return {'snr':float(np.mean([r['snr'] for r in rs])),'stoi':float(np.mean([r['stoi'] for r in rs])),'wer':100*sum(r['errors'] for r in rs)/sum(r['nwords'] for r in rs),'errors':sum(r['errors'] for r in rs),'nwords':sum(r['nwords'] for r in rs),'substitutions':sum(r['substitutions'] for r in rs),'deletions':sum(r['deletions'] for r in rs),'insertions':sum(r['insertions'] for r in rs),'n':len(rs),'utterance_wer_median':float(np.median([100*r['errors']/r['nwords'] for r in rs])),'over100':sum(r['errors']>r['nwords'] for r in rs)}

def main():
 rows=load();manifest=json.loads((ROOT/'data/manifest.json').read_text());test=[r for r in rows if r['method']!='clean'];lookup={(r['id'],r['noise'],r['input_snr'],r['method']):r for r in test}
 summary={'overall':{m:aggregate([r for r in rows if r['method']==m]) for m in METHODS+['clean']},'by_snr':{},'by_noise':{},'paired':{},'by_speaker':{},'metadata':{'n_test_clips':sum(r['split']=='test' for r in manifest),'n_test_speakers':len(set(r['speaker_id'] for r in manifest if r['split']=='test'))}}
 for level in [-5,0,5,10]:summary['by_snr'][str(level)]={m:aggregate([r for r in test if r['method']==m and r['input_snr']==level]) for m in METHODS}
 for noise in ['white','traffic','cafeteria']:summary['by_noise'][noise]={m:aggregate([r for r in test if r['method']==m and r['noise']==noise]) for m in METHODS}
 speakers=sorted(set(r['speaker_id'] for r in test))
 for sp in speakers:summary['by_speaker'][str(sp)]={m:aggregate([r for r in test if r['method']==m and r['speaker_id']==sp]) for m in METHODS}
 def compare(m,ref):
  pairs=[(r,lookup[(r['id'],r['noise'],r['input_snr'],ref)]) for r in test if r['method']==m]
  ds=np.array([a['snr']-b['snr'] for a,b in pairs]);dt=np.array([a['stoi']-b['stoi'] for a,b in pairs]);dw=np.array([100*(a['errors']-b['errors'])/a['nwords'] for a,b in pairs]);rng=np.random.default_rng(5305);boot=[]
  group={sp:[p for p in pairs if p[0]['speaker_id']==sp] for sp in speakers}
  for _ in range(4000):
   samples=[p for sp in rng.choice(speakers,len(speakers),replace=True) for p in group[sp]]
   boot.append([np.mean([a['snr']-b['snr'] for a,b in samples]),np.mean([a['stoi']-b['stoi'] for a,b in samples]),100*sum(a['errors']-b['errors'] for a,b in samples)/sum(a['nwords'] for a,b in samples)])
  # Remove the same pre-identified largest single error-excess condition in both.
  worst=max(pairs,key=lambda p:p[0]['errors']-p[1]['errors']);kept=[p for p in pairs if p is not worst]
  return {'n':len(pairs),'snr_up':int(sum(ds>0)),'stoi_up':int(sum(dt>0)),'wer_down':int(sum(dw<0)),'wer_up':int(sum(dw>0)),'wer_ties':int(sum(dw==0)),'snr_up_wer_worse':int(sum((ds>0)&(dw>0))),'snr_up_stoi_down':int(sum((ds>0)&(dt<0))),'rho_snr_wer':float(spearmanr(ds,dw).statistic),'rho_stoi_wer':float(spearmanr(dt,dw).statistic),'ci':dict(zip(['snr','stoi','wer'],np.percentile(boot,[2.5,97.5],axis=0).T.tolist())),'wer_delta':100*sum(a['errors']-b['errors'] for a,b in pairs)/sum(a['nwords'] for a,b in pairs),'speakers_wer_better':sum(sum(a['errors']-b['errors'] for a,b in group[sp])<0 for sp in speakers),'sensitivity':{'removed_id':worst[0]['id'],'noise':worst[0]['noise'],'snr':worst[0]['input_snr'],'delta_without_largest':100*sum(a['errors']-b['errors'] for a,b in kept)/sum(a['nwords'] for a,b in kept)}}
 for m in METHODS[1:]:summary['paired'][m]=compare(m,'noisy')
 summary['selected_vs_baseline']=compare('selected','baseline')
 # Result panels; SNR/STOI/recognition targets are deliberately kept separate.
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(11.5,3.4))
 for m in METHODS:
  for ax,key in zip(axes,['snr','stoi','wer']):ax.plot([-5,0,5,10],[summary['by_snr'][str(l)][m][key] for l in [-5,0,5,10]],'-o',label=LABELS[m],color=COLORS[m],ms=4)
 for ax,ylabel in zip(axes,['Output SNR (dB)','STOI','Corpus WER (%)']):ax.set(xlabel='Input SNR (dB)',ylabel=ylabel);ax.grid(alpha=.18)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'performance_curves.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11.5,3.3),sharey=True)
 for ax,noise in zip(axes,['white','traffic','cafeteria']):
  for m in METHODS:ax.plot([-5,0,5,10],[aggregate([r for r in test if r['method']==m and r['noise']==noise and r['input_snr']==level])['wer'] for level in [-5,0,5,10]],'-o',label=LABELS[m],color=COLORS[m],ms=4)
  ax.set(title=noise.capitalize(),xlabel='Input SNR (dB)');ax.grid(alpha=.18)
 axes[0].set_ylabel('Corpus WER (%)');axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'wer_by_environment.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10.5,3.5))
 for m in ['baseline','selected']:
  rs=[r for r in test if r['method']==m];b=[lookup[(r['id'],r['noise'],r['input_snr'],'noisy')] for r in rs];dw=[100*(a['errors']-v['errors'])/a['nwords'] for a,v in zip(rs,b)]
  for ax,key,xlabel in zip(axes,['snr','stoi'],['SNR change (dB)','STOI change']):ax.scatter([a[key]-v[key] for a,v in zip(rs,b)],dw,s=12,alpha=.55,color=COLORS[m],label=LABELS[m]);ax.set(xlabel=xlabel,ylabel='Utterance WER change (pp)');ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'paired_relationships.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10.5,3.4));ix=np.arange(len(speakers));width=.25
 for j,m in enumerate(['noisy','baseline','selected']):ax.bar(ix+(j-1)*width,[summary['by_speaker'][str(sp)][m]['wer'] for sp in speakers],width,label=LABELS[m],color=COLORS[m])
 ax.set(xticks=ix,xticklabels=speakers,xlabel='Test speaker ID',ylabel='Corpus WER (%)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'speaker_variation.png',dpi=200);plt.close(fig)
 example=max([r for r in test if r['method']=='baseline'],key=lambda r:r['errors']-lookup[(r['id'],r['noise'],r['input_snr'],'noisy')]['errors'])
 summary['failure_example']={m:lookup[(example['id'],example['noise'],example['input_snr'],m)] for m in METHODS}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['overall'],indent=2));print('Selected-versus-baseline',summary['selected_vs_baseline'])
if __name__=='__main__':main()
