# Speech Enhancement versus Speech Recognition
## Evaluating STFT Wiener Filtering Under Noise

**ELEC5305 Final Project — Xinyu Shen | SID 530137427**

## Project links

- [Project website](https://xinyu-shen-334.github.io/elec5305-project-530537427/)
- [Final report](<ELEC5305_Report_Xinyu_Shen (3).pdf>)
- [Final presentation video](<ELEC5305_Final_Presentation_Xinyu_Shen (1).mp4>)
- [Audio listening page](https://xinyu-shen-334.github.io/elec5305-project-530537427/audio/)
- [Source code](src/)
- [Experimental results](results/)

## Research question

Does STFT Wiener filtering improve speech intelligibility and
recognition accuracy as consistently as it improves waveform SNR?
How do noise estimation and the minimum gain floor affect these outcomes?

## Method

The system uses 16 kHz audio, a 512-sample Hann STFT and a
128-sample hop. Speech power is estimated by non-negative spectral
subtraction. Wiener gains are applied with a minimum floor,
while retaining the noisy phase.

The baseline floor is 0.05. Four candidates were evaluated on the
development set: 0.05, 0.10, 0.20 and 0.35.
The selected floor, 0.35, was frozen before final-test evaluation.

A fixed pretrained Whisper tiny.en recognizer measures word error
rate (WER). No recognition model is trained.
SNR and STOI measure signal quality and objective intelligibility.

## Experimental design

- Development: 4 speech clips from 4 speakers, 48 mixtures.
- Test: 20 speech clips from 10 disjoint speakers, 240 mixtures.
- Test speakers: 5 female and 5 male, 2 clips per speaker.
- Speech source: LibriSpeech test-clean.
- Noise: white Gaussian, DEMAND traffic and DEMAND cafeteria.
- Input SNR: −5, 0, 5 and 10 dB.
- Recorded-noise development interval: 0–60 seconds.
- Recorded-noise test interval: 120–300 seconds.

The project development and test speech subsets both come from
LibriSpeech test-clean; they are not the official benchmark partitions.
Selection uses speaker IDs, sex and clip duration, not recognition scores.

Noise-only calibration immediately precedes the noise added to speech.
The intervals do not overlap and use the same mixing scale.
The study does not claim evaluation on unseen noise recordings.

## Final test results

SNR and STOI are averaged across mixtures.
WER is pooled across reference words.

| Condition | SNR (dB) | STOI | WER (%) |
|---|---:|---:|---:|
| Noisy input | 2.50 | 0.7945 | 28.28 |
| Baseline floor 0.05 | 7.24 | 0.7984 | 29.90 |
| Selected floor 0.35 | 6.56 | 0.8019 | 25.15 |
| Oracle noise PSD | 7.44 | 0.8003 | 30.39 |

Higher SNR does not guarantee lower WER. The selected filter has
lower SNR than the baseline but lower recognition error.

The aggregate WER improvement is driven by white noise.
For recorded traffic and cafeteria noise, the selected filter
remains slightly worse than the noisy input.
The oracle uses actual noise information and is a diagnostic reference.

See the report for detailed results, speaker-level analysis,
uncertainty estimates and limitations.

## Installation and execution

Download or clone this repository and open a terminal in its root folder.
Use Python 3.12 and a virtual environment.

Create the environment:

```sh
python -m venv .venv
```

Activate it on Windows:

```sh
.venv\Scripts\activate
```

Or activate it on macOS/Linux:

```sh
source .venv/bin/activate
```

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

Run the quick DSP demonstration:

```sh
python src/demo.py
```

Run the experiment, analysis and audio export:

```sh
python src/experiment.py
python src/analyse.py
python src/make_audio.py
```

Input speech and noise files are in data/.
The first recognition run downloads the pretrained tiny.en model.
No GPU or model training is required.
The full experiment takes longer than the quick DSP demonstration.

Run settings and package versions are recorded in
results/run_metadata.json.

The experiment can resume existing JSONL records.
For a fresh reproduction, back up results/ first and remove any
development_records.jsonl and test_records.jsonl files before running.
Use a separate output directory when changing experimental settings.

## Repository contents

- src/experiment.py — DSP processing and fixed ASR evaluation.
- src/analyse.py — statistical analysis and result figures.
- src/make_audio.py — audio exports and acoustic figures.
- src/demo.py — quick DSP demonstration.
- data/clean/ — input speech clips.
- data/noise/ — recorded noise files.
- data/manifest.json — clip IDs, transcripts and split membership.
- data/sources.json — data sources, hashes and selection rules.
- results/ — measurements, metadata, summaries and figures.
- audio/ — listening page and four groups of comparison audio.
- index.html — final project homepage.
- requirements.txt — Python dependencies.

The report and presentation are linked at the top of this README.
Earlier proposal and progress files are retained as historical material.

## Listening instructions

Open the audio listening page linked above.

Each comparison plays:
clean → noisy → baseline → selected,
with one-second gaps.

All versions within an example share the same gain.
Keep your playback volume fixed when comparing them.
The four examples were selected by ID, not by best improvement.
The selected floor targets development WER, not listener preference.

Some uploaded audio filenames contain download-generated suffixes.
The listening page maps to those uploaded names.
Running make_audio.py regenerates audio using the original filenames;
the listening-page paths must match whichever filenames are deployed.

## Data attribution

Speech:
Panayotov et al. (2015), LibriSpeech, CC BY 4.0.
https://www.openslr.org/12

Recorded noise:
Thiemann, Ito and Vincent (2013), DEMAND, CC BY-SA 3.0.
https://zenodo.org/records/1227121

Original files retain their source licenses.
Generated mixtures and processed examples are provided under
CC BY-SA 3.0 with both source attributions.
Processing changes are described in the report.
No third-party model weights are redistributed.

## AI assistance acknowledgement

Generative AI assisted code, analysis and report preparation.
Numerical findings come from executed experiments.
The final presentation uses student-provided narration.
