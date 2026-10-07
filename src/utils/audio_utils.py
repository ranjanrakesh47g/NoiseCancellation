import numpy as np
import librosa
from pydub import AudioSegment

CLIPS = {
    "Bird chirp": "noise/bird.wav",
    "Train": "noise/train.wav",
    "Traffic": "noise/traffic.wav",
    "Factory machine": "noise/machine.wav",
}

def wave(samples):
    waveform = np.asarray(samples, np.float32)
    waveform = librosa.to_mono(waveform.T if waveform.ndim > 1 else waveform)
    return waveform / 32768 if waveform.size and np.max(np.abs(waveform)) > 1.5 else waveform

def audio_segment(samples, sr):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    return AudioSegment(pcm.tobytes(), frame_rate=int(sr), sample_width=2, channels=1)

def mix(audio, snr_db, *flags):
    if audio is None:
        return None
    sr, samples = audio
    speech = wave(samples)
    if speech.size == 0:
        return None
    mixed = audio_segment(speech, sr)
    speech_db = mixed.dBFS
    for noise_path, selected in zip(CLIPS.values(), flags):
        if not selected:
            continue
        noise, _ = librosa.load(noise_path, sr=sr, mono=True)
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))[:len(speech)]
        clip = audio_segment(noise, sr)
        gain_db = speech_db - clip.dBFS - float(snr_db)
        mixed = mixed.overlay(clip.apply_gain(gain_db))
    return mixed.frame_rate, np.array(mixed.get_array_of_samples(), dtype=np.int16)
