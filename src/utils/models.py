import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import pipeline
import librosa
from src.utils.audio_utils import wave

ASR_MODELS = {
    "whisper-small": "distil-whisper/distil-small.en",
    "whisper-large": "openai/whisper-large-v3",
    "parakeet": "ai-and-i-project/parakeet-tdt-0.6b-v2-hf",
}

class Models:
    def __init__(self):
        self.asrs = [
            pipeline(task="automatic-speech-recognition",
                     model=model_id,
                     model_kwargs={"cache_dir": "models"})
            for model_id in ASR_MODELS.values()
        ]
        for name, asr in zip(ASR_MODELS, self.asrs):
            print("asr_freq:", name, asr.feature_extractor.sampling_rate)

    def transcribe(self, audio, asr):
        if audio is None:
            return ""
        sr, samples = audio
        rate = asr.feature_extractor.sampling_rate
        waveform = librosa.resample(wave(samples), orig_sr=sr, target_sr=rate)
        return asr({"array": waveform, "sampling_rate": rate})["text"]

    def transcribe_all(self, audio):
        return [self.transcribe(audio, asr) for asr in self.asrs]
