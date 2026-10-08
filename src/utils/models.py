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
    def __init__(self, name=None):
        ids = {name: ASR_MODELS[name]} if name else ASR_MODELS
        self.names = list(ids)
        self.asrs = [
            pipeline(task="automatic-speech-recognition",
                     model=model_id,
                     model_kwargs={"cache_dir": "models"})
            for model_id in ids.values()
        ]
        for model_name, asr in zip(self.names, self.asrs):
            if model_name.startswith("whisper"):
                asr.generation_config.condition_on_prev_tokens = False
            if model_name == "whisper-large":
                asr.generation_config.language = "en"
            print("asr_freq:", model_name, asr.feature_extractor.sampling_rate)

    def _prepare(self, audio, asr):
        if audio is None:
            return None
        sr, samples = audio
        rate = asr.feature_extractor.sampling_rate
        waveform = librosa.resample(wave(samples), orig_sr=sr, target_sr=rate)
        waveform, _ = librosa.effects.trim(waveform)
        if waveform.size == 0:
            return None
        return {"array": waveform, "sampling_rate": rate}

    def transcribe(self, audio, asr):
        prepared = self._prepare(audio, asr)
        if prepared is None:
            return ""
        return asr(prepared)["text"]

    def transcribe_all(self, audio):
        return [self.transcribe(audio, asr) for asr in self.asrs]
