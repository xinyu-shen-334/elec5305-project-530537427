"""Reproducible STFT Wiener filtering progress demo.

The default synthetic run is a pipeline check, NOT a speech or ASR result.
For WAV tests, --noise supplies noise-only calibration samples. An oracle PSD
uses the actual mixed-in noise; practical PSD uses an independent noise segment.
"""
import argparse
from math import gcd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, resample_poly, stft

FS, NFFT, HOP, EPS = 16000, 512, 128, 1e-12


def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.astype(np.float64).mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float64) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float64)
    if sr != FS:
        divisor = gcd(sr, FS)
        x = resample_poly(x, FS // divisor, sr // divisor)
    if len(x) < NFFT or not np.all(np.isfinite(x)) or np.mean(x*x) < EPS:
        raise ValueError(f'Invalid, too short, or silent audio: {path}')
    return x


def mix_snr(clean, noise, target_snr, rng):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(clean) + 1))
    n = noise[start:start+len(clean)].copy()
    n -= np.mean(n)
    power = np.mean(n*n)
    if power < EPS:
        raise ValueError('Noise calibration source is silent')
    n *= np.sqrt(np.mean(clean*clean) / (power * 10 ** (target_snr/10)))
    return clean + n, n


def spectrum(x):
    return stft(x, fs=FS, window='hann', nperseg=NFFT,
                noverlap=NFFT-HOP, nfft=NFFT, boundary='zeros', padded=True)


def noise_psd_from_signal(noise):
    _, _, z = spectrum(noise)
    return np.mean(np.abs(z)**2, axis=1)


def enhance(noisy, noise_psd):
    _, _, y = spectrum(noisy)
    noise_psd = np.asarray(noise_psd).reshape(-1, 1)
    speech_psd = np.maximum(np.abs(y)**2 - noise_psd, 0)
    gain = speech_psd / (speech_psd + noise_psd + EPS)
    gain = np.maximum(gain, 0.05)
    _, output = istft(gain*y, fs=FS, window='hann', nperseg=NFFT,
                      noverlap=NFFT-HOP, nfft=NFFT, boundary=True)
    return output[:len(noisy)], gain


def snr_db(clean, signal):
    e = clean - signal[:len(clean)]
    return 10*np.log10((np.mean(clean**2)+EPS) / (np.mean(e**2)+EPS))


def plot_result(clean, noisy, practical, oracle, gains, out):
    ts = np.arange(len(clean)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ['Clean reference', 'Noisy input', 'Practical PSD enhanced', 'Oracle PSD enhanced']
    for ax, y, lab in zip(axes, [clean, noisy, practical, oracle], labels):
        ax.plot(ts, y, lw=0.55); ax.set(title=lab, ylabel='Amplitude')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(out/'waveforms.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, y, lab in zip(axes, [clean, noisy, practical, oracle], labels):
        f, t, z = spectrum(y)
        ax.pcolormesh(t, f, 20*np.log10(np.maximum(abs(z), 1e-7)),
                      shading='auto', vmin=-90, vmax=-15)
        ax.set(title=lab, xlabel='Time (s)', ylim=(0, 6000))
    axes[0].set_ylabel('Frequency (Hz)')
    fig.tight_layout(); fig.savefig(out/'spectrograms.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, gain, lab in zip(axes, gains, ['Practical PSD gain', 'Oracle PSD gain']):
        ax.imshow(gain, origin='lower', aspect='auto', vmin=0, vmax=1,
                  extent=[0, len(clean)/FS, 0, FS/2])
        ax.set(title=lab, xlabel='Time (s)', ylim=(0, 6000))
    axes[0].set_ylabel('Frequency (Hz)')
    fig.tight_layout(); fig.savefig(out/'wiener_gain.png', dpi=140); plt.close(fig)


def save_wav(path, signal, scale):
    wavfile.write(path, FS, (np.clip(signal/scale, -1, 1)*32767).astype(np.int16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path)
    parser.add_argument('--noise', type=Path)
    parser.add_argument('--snr', type=float, default=0.0)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if bool(args.clean) != bool(args.noise):
        parser.error('Supply both --clean and --noise, or neither for synthetic demonstration.')
    rng = np.random.default_rng(5305)
    if args.clean:
        clean, source = load_wav(args.clean), load_wav(args.noise)
        label = 'USER-SUPPLIED WAV EXPERIMENT (not independently verified speech data)'
    else:
        t = np.arange(2*FS)/FS
        envelope = np.sin(np.pi*t/2)**2
        clean = envelope * (0.2*np.sin(2*np.pi*180*t) + 0.1*np.sin(2*np.pi*360*t)
                            + 0.05*np.sin(2*np.pi*540*t))
        source = rng.normal(size=8*FS)
        label = 'SYNTHETIC SMOKE TEST (NOT HUMAN SPEECH OR ASR)'
    noisy, added_noise = mix_snr(clean, source, args.snr, rng)
    oracle_psd = noise_psd_from_signal(added_noise)
    # Independent calibration uses the first second of the original noise source,
    # rather than the exact segment added to the speech (practical but imperfect).
    calibration = source[:FS] - np.mean(source[:FS])
    calibration *= np.sqrt(np.mean(added_noise**2)/(np.mean(calibration**2)+EPS))
    practical_psd = noise_psd_from_signal(calibration)
    practical, practical_gain = enhance(noisy, practical_psd)
    oracle, oracle_gain = enhance(noisy, oracle_psd)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scale = max(1, *(float(np.max(np.abs(x))) for x in (clean, noisy, practical, oracle)))
    for name, signal in [('clean', clean), ('noisy', noisy),
                         ('practical_enhanced', practical), ('oracle_enhanced', oracle)]:
        save_wav(out/f'{name}.wav', signal, scale)
    plot_result(clean, noisy, practical, oracle, (practical_gain, oracle_gain), out)
    input_snr = snr_db(clean, noisy)
    output_practical = snr_db(clean, practical)
    output_oracle = snr_db(clean, oracle)
    report = (f'{label}\nRequested input SNR: {args.snr:.2f} dB\n'
              f'Measured input SNR: {input_snr:.2f} dB\n'
              f'Practical-estimate output SNR: {output_practical:.2f} dB '
              f'(change {output_practical-input_snr:+.2f} dB)\n'
              f'Oracle-estimate output SNR: {output_oracle:.2f} dB '
              f'(change {output_oracle-input_snr:+.2f} dB)\n'
              'One synthetic/demo run, not a benchmark; no STOI/WER measured.\n')
    (out/'demo_metrics.txt').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
