import re
import time
import numpy as np
import librosa
from pydub import AudioSegment

CLIPS = {
    "Claps & Cheers": "noise/claps.wav",
    "Traffic & Wind": "noise/traffic_wind.wav",
    "Retail Ambient Noise": "noise/retail.wav",
    "Bird Chirp": "noise/bird.wav",
}

def wave(samples):
    waveform = np.asarray(samples, np.float32)
    waveform = librosa.to_mono(waveform.T if waveform.ndim > 1 else waveform)
    return waveform / 32768 if waveform.size and np.max(np.abs(waveform)) > 1.5 else waveform

def audio_segment(samples, sr):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    return AudioSegment(pcm.tobytes(), frame_rate=int(sr), sample_width=2, channels=1)

def mix(audio, snr_db, noise):
    if audio is None:
        return None
    sr, samples = audio
    speech = wave(samples)
    if speech.size == 0:
        return None
    mixed = audio_segment(speech, sr)
    speech_db = mixed.dBFS
    clip_wave, _ = librosa.load(CLIPS[noise], sr=sr, mono=True)
    clip_wave = np.tile(clip_wave, int(np.ceil(len(speech) / len(clip_wave))))[:len(speech)]
    clip = audio_segment(clip_wave, sr)
    gain_db = speech_db - clip.dBFS - float(snr_db)
    mixed = mixed.overlay(clip.apply_gain(gain_db))
    return mixed.frame_rate, np.array(mixed.get_array_of_samples(), dtype=np.int16)

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]

_PUNCT = re.compile(r"[^\w\s]|_")

def comparable(text):
    return _PUNCT.sub("", text.lower())

def transcript_limit(original):
    n = len(original)
    return 0 if n <= 2 else 1 if n <= 5 else 2 if n <= 10 else 3

def match_snr(audio, noise, original_transcript, asr, transcribe):
    lo, hi, best, text = -10, 20, 20, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        noisy_transcript = transcribe(mix(audio, mid, noise), asr)
        noisy, original = comparable(noisy_transcript), comparable(original_transcript)
        if levenshtein(noisy, original) <= transcript_limit(original):
            best, text = mid, noisy_transcript
            hi = mid - 1
        else:
            if mid == 20:
                text = noisy_transcript
            lo = mid + 1
    return best, text

def transcribe_mix(audio, noise, snr_db, asr, transcribe):
    noisy = mix(audio, snr_db, noise)
    return noisy, transcribe(noisy, asr)

def match_all(audio, noise, originals, asrs, transcribe):
    from src.utils.logging import latency_ms
    found, latencies = [], []
    for original, asr in zip(originals, asrs):
        start = time.perf_counter()
        found.append(match_snr(audio, noise, original, asr, transcribe))
        latencies.append(latency_ms(start))
    snrs, texts = zip(*found)
    return [*snrs, mix(audio, snrs[0], noise), *texts], latencies
