from transformers import pipeline
import sys
import types
import numpy as np
import librosa
import torch
import torchaudio

sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
sys.modules["torchaudio.backend.common"] = types.ModuleType("torchaudio.backend.common")
sys.modules["torchaudio.backend.common"].AudioMetaData = object
from df.enhance import enhance, init_df

from src.utils.audio_utils import wave

class Models:
    def __init__(self):
        self.asr = pipeline(task="automatic-speech-recognition",
                            model="distil-whisper/distil-small.en",
                            model_kwargs={"cache_dir": "models"})
        print("asr_freq:", self.asr.feature_extractor.sampling_rate)
        self.df_model, self.df_state, _ = init_df(log_file=None)

    def transcribe(self, audio):
        if audio is None:
            return ""
        sr, samples = audio
        rate = self.asr.feature_extractor.sampling_rate
        waveform = librosa.resample(wave(samples), orig_sr=sr, target_sr=rate)
        return self.asr({"array": waveform, "sampling_rate": rate})["text"]

    def denoise(self, audio):
        if audio is None:
            return None
        sr, samples = audio
        resampled = librosa.resample(wave(samples), orig_sr=sr, target_sr=self.df_state.sr())
        denoised = enhance(self.df_model, self.df_state, torch.from_numpy(np.ascontiguousarray(resampled)).unsqueeze(0)).squeeze(0).numpy()
        return self.df_state.sr(), (np.clip(denoised, -1, 1) * 32767).astype(np.int16)
