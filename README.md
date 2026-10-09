# Speech Enhancement versus Speech Recognition: Evaluating STFT Wiener Filtering Under Noise

**ELEC5305 progress submission — Xinyu Shen (SID 530537427)**

**GitHub Pages:** https://xinyu-shen-334.github.io/elec5305-project-530537427/  
**GitHub Repository:** https://github.com/xinyu-shen-334/elec5305-project-530537427

## Revised research question

Does STFT Wiener filtering improve intelligibility and automatic speech recognition as consistently as it improves waveform SNR under different noise environments? To what extent does performance depend on noise PSD estimation accuracy?

## Why this is meaningful

Denoising and speech recognition are different objectives: suppression of weak spectral components may remove useful phonetic features. We will compare input/output SNR, objective intelligibility (STOI), and Whisper word error rate (WER) on exactly the same audio. A positive SNR change is not assumed to mean a positive change in recognition accuracy.

## Work completed for this progress checkpoint

- Python STFT / inverse-STFT with a 16 kHz sample rate, Hann window, 512-point FFT and 128-sample hop.
- Controlled additive noise mixing at user-selected input SNR.
- Explicit Wiener gain using the mixture spectrum and a noise PSD estimate, with a gain floor of 0.05.
- **Practical noise PSD** estimated from an independent noise-only calibration segment, and **oracle PSD** estimated using the exact additive noise as an experimental control.
- Reconstruction, audio export, waveform / spectrogram / Wiener-gain figures and waveform-error SNR measurements.
- Reproducible *synthetic* demonstration (seed 5305) requiring no downloaded datasets. This verifies the pipeline but **is not real-speech evidence**.

**Not yet completed:** Librispeech/DEMAND experimental results, speech-based STOI metrics, Whisper transcription, WER and correlation analysis. The code and plots in `results/` must not be represented as human-speech recognition results. A practical PSD from a calibration segment is still simplified, and can be inaccurate for time-varying noise.

## Folder structure

```text
README.md                       project, methods, limitations and next steps
index.html                      GitHub Pages project status (optional home page)
PROGRESS_SUBMISSION.txt         short text for the course submission box
requirements.txt                minimal Python dependencies
src/wiener_experiment.py        fully reproducible DSP experiment
results/*.png                   generated synthetic test plots
results/*.wav                   generated synthetic test audio
results/demo_metrics.txt        preliminary synthetic SNR figures
```

## Installation and running

Python 3.10+ recommended.

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/wiener_experiment.py --output results
```

Run with your own real WAV files (clean speech + environmental noise):

```powershell
python src/wiener_experiment.py --clean data/clean.wav --noise data/noise.wav --snr 0 --output results/real_test
```

The WAV files can be mono or stereo (stereo is averaged to mono) and are resampled to 16 kHz. Do not upload large third-party datasets to GitHub. Experimental results depend on the chosen data and are not guaranteed to improve.

## Mathematical implementation

Mixture: `y[n] = s[n] + v[n]`. The STFT gives `Y(k,m)`. The noise PSD `P_v(k)` is calculated from a set of noise-only frames; the estimated speech PSD is `max(|Y(k,m)|² - P_v(k), 0)`. The Wiener gain is:

`G(k,m) = max(0.05, P_s(k,m) / (P_s(k,m) + P_v(k) + eps))`.

The filtered STFT is `X_hat(k,m) = G(k,m)Y(k,m)`, using noisy phase; inverse STFT reconstructs the waveform. Input/output SNR use the clean reference and waveform difference. **Oracle** noise PSD is computed from the exact added noise. **Practical** noise PSD uses another calibration segment from the same noise recording and therefore has imperfect finite-sample statistics.

## Next research milestones

1. Select 20–50 transcribed LibriSpeech `dev-clean` utterances and establish baseline Whisper WER, keeping model and decoding settings fixed.
2. Choose three contrasting DEMAND environmental noise recordings and mix at **−5, 0, +5, +10 dB**.
3. For clean, noisy, practically enhanced and oracle-enhanced audio, compute SNR and STOI; run the same pretrained Whisper recogniser, normalize transcripts, and calculate WER.
4. Aggregate by noise type, SNR and estimator; plot SNR/STOI change versus WER change and investigate cases where higher SNR does not improve WER.
5. Examine Wiener gain maps and selected clean/noisy/enhanced log-Mel representations to interpret the results.

No training or fine-tuning of Whisper is needed; the learned model is a fixed evaluation instrument, not the DSP implementation.

## Data and resources

- [LibriSpeech speech corpus](https://www.openslr.org/12)
- [DEMAND environmental noise](https://zenodo.org/records/1227121)
- [Whisper pretrained recogniser](https://github.com/openai/whisper)
- [pystoi metric](https://github.com/mpariente/pystoi)
- [JiWER WER metric](https://github.com/jitsi/jiwer)

## Related literature

1. Lim, J. S., & Oppenheim, A. V. (1979). Enhancement and bandwidth compression of noisy speech. *Proceedings of the IEEE*, 67(12), 1586–1604.
2. Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 32(6), 1109–1121.
3. Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(2), 113–120.
4. Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An algorithm for intelligibility prediction of time-frequency weighted noisy speech. *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7), 2125–2136.
5. Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. *ICASSP*, 5206–5210.
6. Thiemann, J., Ito, N., & Vincent, E. (2013). The diverse environments multi-channel acoustic noise database (DEMAND). *Meetings on Acoustics*, 19, 035081.
7. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *ICML*, 28492–28518.

## Submission status

**Initial DSP implementation and demonstrations complete.** Real-speech and downstream-ASR research are planned and not yet claimed as completed. These are early progress materials for teaching feedback, not a final report.
