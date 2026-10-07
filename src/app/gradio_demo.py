import os
import gradio as gr

from src.utils.audio_utils import CLIPS, mix
from src.utils.models import ASR_MODELS, Models

class Demo:
    def __init__(self):
        self.models = None
        self.app = None

    def refresh(self, audio, snr_db, *flags):
        noisy = mix(audio, snr_db, *flags)
        return [item for text in self.models.transcribe_all(noisy) for item in (noisy, text)]

    def on_audio(self, audio, snr_db, *flags):
        return self.models.transcribe(audio, self.models.asrs[0]), *self.refresh(audio, snr_db, *flags)

    def panel(self, source):
        audio = gr.Audio(sources=source, type="numpy", label="Clean speech")
        clear = gr.Button("Clear")
        transcript = gr.Textbox(label="Transcription", lines=3)
        with gr.Row():
            boxes = [gr.Checkbox(True, label=label) for label in CLIPS]
        snr = gr.Slider(-10, 20, value=5, step=1, label="SNR in dB (lower = louder noise)")
        model_outputs = []
        for model_id in ASR_MODELS:
            with gr.Row():
                with gr.Column():
                    model_outputs.append(gr.Audio(label=model_id))
                with gr.Column():
                    model_outputs.append(gr.Textbox(label=f"{model_id} transcription", lines=3))
        inputs = [audio, snr, *boxes]
        clear.click(lambda: (None, "", 5, *([None, ""] * len(ASR_MODELS))), outputs=[audio, transcript, snr, *model_outputs])
        event = audio.stop_recording if source == "microphone" else audio.upload
        event(self.on_audio, inputs, [transcript, *model_outputs])
        snr.input(self.refresh, inputs, model_outputs, show_progress="hidden", trigger_mode="always_last")
        for box in boxes:
            box.change(self.refresh, inputs, model_outputs, show_progress="hidden")

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
        self.app.launch(server_port=server_port)

if __name__ == "__main__":
    Demo().launch()
