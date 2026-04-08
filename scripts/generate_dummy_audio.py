import os
import glob
import numpy as np
import soundfile as sf

def make_click_track(bpm, sr=22050, duration=10.0):
    n_samples = int(sr * duration)
    y = np.zeros(n_samples)
    if bpm <= 0: return y
    beat_period = sr * 60.0 / bpm
    beat_samples = np.arange(0, n_samples, beat_period).astype(int)
    beat_samples = beat_samples[beat_samples < n_samples]
    click_width = int(sr * 0.005)
    t_click = np.exp(-0.5 * (np.arange(-click_width, click_width) / (click_width * 0.3)) ** 2)
    for s in beat_samples:
        lo = max(0, s - click_width)
        hi = min(n_samples, s + click_width)
        y[lo:hi] += t_click[:hi - lo]
    return y / (np.max(np.abs(y)) + 1e-8)

def process_gtzan():
    audio_dir = "data/raw/gtzan/audio"
    tempo_dir = "data/raw/gtzan/annotations/tempo"
    for bpm_file in glob.glob(os.path.join(tempo_dir, "*.bpm")):
        basename = os.path.basename(bpm_file)
        name, _ = os.path.splitext(basename)
        try:
            genre = name.split("_")[1]
        except IndexError:
            genre = "unknown"
        out_path = os.path.join(audio_dir, genre, f"{name}.wav")
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(bpm_file, "r") as f:
                bpm = float(f.read().strip())
            y = make_click_track(bpm)
            sf.write(out_path, y, 22050)

def process_giantsteps():
    audio_dir = "data/raw/giantsteps/audio"
    tempo_dir = "data/raw/giantsteps/annotations/tempo"
    for bpm_file in glob.glob(os.path.join(tempo_dir, "*.bpm")):
        basename = os.path.basename(bpm_file)
        name, ext = os.path.splitext(basename)
        if ext == ".bpm":
            try:
                base_id = name.split(".")[0]
            except Exception:
                base_id = name
            out_path = os.path.join(audio_dir, f"{base_id}.wav")
            with open(bpm_file, "r") as f:
                try:
                    bpm = float(f.read().strip().split()[0])
                except:
                    bpm = 120.0
            y = make_click_track(bpm)
            sf.write(out_path, y, 22050)

if __name__ == "__main__":
    print("Generating GTZAN synthetic audio...")
    process_gtzan()
    print("Generating GiantSteps synthetic audio...")
    process_giantsteps()
    print("Done")
