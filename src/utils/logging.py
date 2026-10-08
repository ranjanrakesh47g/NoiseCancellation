import json
import os
import time
import wave
from datetime import datetime

from src.utils.audio_utils import wave as mono
from src.utils.models import ASR_MODELS

os.makedirs("logs/audio", exist_ok=True)

def save_wav(audio, label, counted=False, directory="logs/audio"):
    if audio is None:
        return ""
    sr, samples = audio
    pcm = samples
    if str(getattr(samples, "dtype", "")) != "int16" or getattr(samples, "ndim", 1) != 1:
        pcm = (mono(samples).clip(-1, 1) * 32767).astype("int16")
    os.makedirs(directory, exist_ok=True)
    if counted:
        safe = " ".join(str(label).replace("/", " ").split()).rstrip(".")[:120].rstrip() or "audio"
        path, count = f"{directory}/{safe}.wav", 2
        while os.path.exists(path):
            path, count = f"{directory}/{safe} {count}.wav", count + 1
    else:
        path = f"{directory}/{label}.wav"
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(pcm.tobytes())
    return path

def latency_ms(start):
    return f"{round((time.perf_counter() - start) * 1000)} ms"

def log_result(noise, result, audio_path, originals, latencies):
    n = len(ASR_MODELS)
    snrs, noisy, texts = result[:n], result[n], result[n + 1:]
    name = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else "audio"
    noisy_path = save_wav(noisy, f"{name}_{noise}")
    latencies = [*latencies, *([None] * n)][:n]
    record = {
        "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "audio_path": audio_path,
        "noise-type": noise,
        "noisy-audio-path": noisy_path,
        "results": {
            name: {
                "original_transcript": original,
                "latency": latency,
                "optimal_snr_db": snr,
                "reconstructed_transcript": text,
            }
            for name, original, latency, snr, text in zip(ASR_MODELS, originals, latencies, snrs, texts)
        },
    }
    with open("logs/transcript.log", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n\n")
