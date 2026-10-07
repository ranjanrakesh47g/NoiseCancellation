import os
import time
import gradio as gr
from src.utils.audio_utils import CLIPS, match_all, transcribe_mix
from src.utils.logging import latency_ms, log_result, save_wav
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

class Demo:
    def __init__(self):
        self.models = None
        self.app = None
        self.audio_path = ""
        self.latencies = []

    def on_noise(self, audio, noise, *originals):
        result, self.noise_latencies = match_all(audio, noise, originals, self.models.asrs, self.models.transcribe)
        log_result(noise, result, self.audio_path, originals, self.noise_latencies)
        return result

    def on_audio(self, audio, noise):
        originals, self.latencies = [], []
        for asr in self.models.asrs:
            start = time.perf_counter()
            originals.append(self.models.transcribe(audio, asr))
            self.latencies.append(latency_ms(start))
        self.audio_path = save_wav(audio, originals[0] if originals else "", counted=True)
        result, _ = match_all(audio, noise, originals, self.models.asrs, self.models.transcribe)
        log_result(noise, result, self.audio_path, originals, self.latencies)
        return [*originals, *result]

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
        noise.change(self.on_noise, [audio, noise, *clean_texts], [*snrs, noisy, *noisy_texts], show_progress="hidden")
        for index, snr in enumerate(snrs):
            snr.input(lambda audio, noise, snr_db, index=index: transcribe_mix(audio, noise, snr_db, self.models.asrs[index], self.models.transcribe), [audio, noise, snr], [noisy, noisy_texts[index]], show_progress="hidden", trigger_mode="always_last")

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
