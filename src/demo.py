"""Fast DSP demonstration (no ASR inference or retraining).
python src/demo.py
Writes a fresh, jointly scaled clean/noisy/baseline/selected audio set.
"""
import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from pystoi import stoi
from experiment import ROOT,FS,load_sources,mixture,psd,enhance,snr

def main():
 t0=time.perf_counter();manifest,noises=load_sources()
 r=next(r for r in manifest if r['split']=='test' and r['sex']=='F')
 floor=json.loads((ROOT/'results/development_selection.json').read_text())['selected_floor']
 s,y,v,cal,_=mixture(r,'traffic',0,noises)
 baseline,_=enhance(y,psd(cal),.05);selected,_=enhance(y,psd(cal),floor)
 waves={'clean':s,'noisy':y,'baseline':baseline,'selected':selected};gain=.9/max(float(np.max(abs(x))) for x in waves.values())
 out=ROOT/'demo_output';out.mkdir(exist_ok=True)
 print('STFT Wiener DSP demonstration | 16000 Hz | 512 FFT | hop 128')
 print('Utterance:',r['id'],'| traffic noise | target 0 dB')
 print('Reference:',r['text'])
 print('Independent calibration: preceding 1-second noise-only interval')
 print('Method        Output SNR (dB)       STOI')
 for name,x in waves.items():
  sf.write(out/(name+'.wav'),gain*x,FS,subtype='PCM_24')
  if name!='clean':print(f'{name:<12}  {snr(s,x):>10.3f}           {stoi(s,x,FS,extended=False):.3f}')
 print(f'Selected gain floor: {floor:.2f}')
 print('Saved four WAV files in demo_output/ using one common export gain')
 print(f'DSP and export completed in {time.perf_counter()-t0:.3f} seconds')
if __name__=='__main__':main()
