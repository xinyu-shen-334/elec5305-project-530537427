# Speech Enhancement versus Speech Recognition
Xinyu Shen | SID 530137427 | ELEC5305

## Revised study
The main study replaces the earlier single-speaker synthetic-noise pilot.
Development: 4 clips / 4 speakers, 48 mixtures, four candidate gain floors.
Final test: 20 clips / 10 disjoint speakers (5 F, 5 M), 240 mixtures.
Speech: official LibriSpeech test-clean. These project development and test
partitions are both selected from that archive; they are not official benchmark
partitions. Selection depends on IDs, sex and duration, not ASR scores.
Noise: white Gaussian + DEMAND STRAFFIC/ch01.wav + PCAFETER/ch01.wav.
Development noise time: 0-60s. Test: 120-300s. No claim of unseen environments.
Levels: -5, 0, 5, 10 dB. The calibration interval immediately precedes the noise
added to speech and shares the same scale; its power is not forced to match.

Selected floor: 0.35, frozen before final-test evaluation.
Final corpus WER: noisy 28.28%, baseline 29.90%,
selected 25.15%. SNR/STOI/WER measure different objectives.
The overall WER gain is driven by white noise. The selected filter remains
slightly worse than the noisy input on both real-noise environments.

## Install and run
Use Python 3.12 and a virtual environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/experiment.py
python src/analyse.py
python src/make_audio.py
python src/demo.py  # quick DSP-only demonstration
```

Audio/data files are included. The first ASR run downloads the pinned tiny.en
model (~75 MB). No GPU or model training is required. All run settings and
package versions are recorded in results/run_metadata.json.

The experiment resumes existing JSONL records and rewrites complete CSV files.
To reproduce from scratch, first copy the supplied results folder to a backup,
then remove the two *_records.jsonl files before running. For altered settings,
use a new results directory rather than mixing old and new records. Noisy and
processed outputs use the same decoder safeguards: no repeated three-token n-gram and at
most 96 new tokens. This is why old pilot WER is not directly comparable.

## Contents
- ELEC5305_Report_Xinyu_Shen.pdf: revised research report
- report.md: editable report text
- demo.mp4: explanation, comparison audio and measured findings
- index.html: project website entry point
- audio/index.html: audio listening page
- audio/example_*/: clean, noisy, baseline, selected, oracle and sampler files
- data/manifest.json: clip IDs, transcripts, split membership and hashes
- data/sources.json: source identities, hashes and selection rules
- results/development_selection.json: fixed development selection
- results/development_measurements.csv: all development outputs/transcripts
- results/test_measurements.csv: all final-test outputs/transcripts
- results/summary.json: aggregates, paired counts and speaker bootstrap intervals
- results/*.png: performance, speaker, waveform, spectrum and gain plots
- src/experiment.py: DSP and fixed ASR experiment
- src/analyse.py: statistical and graphical analysis
- src/make_audio.py: jointly scaled exports and acoustic figures

## Fair audio comparison
Each comparison.wav/.mp3 plays CLEAN -> NOISY -> BASELINE -> SELECTED,
with one second of silence after each version. Do not interpret different
independent normalisations as denoising benefits: none are applied here.
Four examples are selected by ID, not by best improvement. The selected floor
optimises development recognition error, not subjective listener preference.

## Data rights and attribution
LibriSpeech: Panayotov et al. (2015), CC BY 4.0.
https://www.openslr.org/12
DEMAND: Thiemann, Ito and Vincent (2013), CC BY-SA 3.0.
https://zenodo.org/records/1227121
Original speech and noise retain their own source licenses. Generated mixtures
and processed examples are provided under CC BY-SA 3.0 with both attributions.
Processing changes are described in the report. No third-party model weights
are redistributed. See the software/model repositories for their licenses.
Generative AI assisted code, analysis and report preparation; all numerical
claims are derived from executed experiments. Review and understand the work
and follow the course acknowledgement requirements.

## Submission
Upload the report PDF and actual demo.mp4 to Canvas. Publish this folder's
CONTENTS to the project repository, preserving the folder structure. GitHub
Pages should serve index.html, the PDF, demo.mp4 and audio/index.html.
Project: https://xinyu-shen-334.github.io/elec5305-project-530537427/
Code: https://github.com/xinyu-shen-334/elec5305-project-530537427
Publication and Canvas submission have not been performed by this package.

## Final recorded presentation
The demo.mp4 now uses the student-provided final narration and recorded page timing. Experimental audio is preserved. A constant +3 dB gain is applied to the complete audio track; no independent normalisation of comparison examples is applied.
