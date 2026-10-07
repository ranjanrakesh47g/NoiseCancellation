import os
import re
import gradio as gr

from src.utils.audio_utils import CLIPS, mix
from src.utils.models import ASR_MODELS, Models

TRANSCRIPTION_CSS = """
.transcription-row > .form { flex-wrap: wrap; background: var(--block-background-fill); }
.transcription-row > .form::before {
    content: "Transcription";
    flex-basis: 100%;
    margin: var(--block-label-margin);
    color: var(--block-title-text-color);
    font-size: var(--block-title-text-size);
    font-weight: var(--block-title-text-weight);
}
.transcription-row > .form > .block { flex: 1; min-width: 0; }
"""

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

class Demo:
    def __init__(self):
        self.models = None
        self.app = None

    def match_snr(self, audio, noise, original_transcript, asr):
        lo, hi, best, text = -10, 20, 20, ""
        while lo <= hi:
            mid = (lo + hi) // 2
            noisy_transcript = self.models.transcribe(mix(audio, mid, noise), asr)
            noisy, original = comparable(noisy_transcript), comparable(original_transcript)
            if levenshtein(noisy, original) <= transcript_limit(original):
                best, text = mid, noisy_transcript
                hi = mid - 1
            else:
                if mid == 20:
                    text = noisy_transcript
                lo = mid + 1
        return best, text

    def refresh_one(self, audio, noise, snr_db, index):
        noisy = mix(audio, snr_db, noise)
        return noisy, self.models.transcribe(noisy, self.models.asrs[index])

    def refresh(self, audio, noise, *originals):
        found = [self.match_snr(audio, noise, original, asr) for original, asr in zip(originals, self.models.asrs)]
        snrs, texts = zip(*found)
        return [*snrs, mix(audio, snrs[0], noise), *texts]

    def on_audio(self, audio, noise):
        originals = self.models.transcribe_all(audio)
        return [*originals, *self.refresh(audio, noise, *originals)]

    def panel(self, source):
        audio = gr.Audio(sources=source, type="numpy", label="Clean speech")
        clear = gr.Button("Clear")
        with gr.Row(elem_classes=["transcription-row"]):
            clean_texts = [gr.Textbox(label=f"{name} transcription", lines=3) for name in ASR_MODELS]
        with gr.Row():
            noise = gr.Radio(list(CLIPS), value=next(iter(CLIPS)), label="Noise-type")
        noisy = gr.Audio(label="Noisy audio")
        snrs, noisy_texts = [], []
        with gr.Row():
            for name in ASR_MODELS:
                with gr.Column(min_width=0):
                    snrs.append(gr.Slider(-10, 20, value=5, step=1, min_width=0, label="SNR in dB (lower = louder noise)"))
                    noisy_texts.append(gr.Textbox(label=f"{name} transcription", lines=3))
        clear.click(lambda: (None, *[""] * len(ASR_MODELS), next(iter(CLIPS)), None, *([5] * len(ASR_MODELS)), *[""] * len(ASR_MODELS)), outputs=[audio, *clean_texts, noise, noisy, *snrs, *noisy_texts])
        event = audio.stop_recording if source == "microphone" else audio.upload
        event(self.on_audio, [audio, noise], [*clean_texts, *snrs, noisy, *noisy_texts])
        noise.change(self.refresh, [audio, noise, *clean_texts], [*snrs, noisy, *noisy_texts], show_progress="hidden")
        for index, snr in enumerate(snrs):
            snr.input(lambda audio, noise, snr_db, index=index: self.refresh_one(audio, noise, snr_db, index), [audio, noise, snr], [noisy, noisy_texts[index]], show_progress="hidden", trigger_mode="always_last")

    def build(self):
        with gr.Blocks() as microphone:
            self.panel("microphone")
        with gr.Blocks() as upload:
            self.panel("upload")
        app = gr.Blocks()
        with app:
            gr.TabbedInterface([microphone, upload], ["Microphone", "Upload Audio File"])
        return app

    def launch(self, server_port=None):
        if self.models is None:
            self.models = Models()
        if self.app is None:
            self.app = self.build()
        if server_port is None:
            server_port = int(os.environ.get("PORT2", 8000))
        self.app.launch(server_port=server_port, css=TRANSCRIPTION_CSS)

if __name__ == "__main__":
    Demo().launch()
