import os
import gradio as gr

from src.utils.audio_utils import mix
from src.utils.logging import save_wav
from src.utils.models import ASR_MODELS, Models

NOISES = ("Claps & Cheers", "Traffic & Wind", "Retail Ambient Noise")
SNRS = (-5, 0, 5, 10)

class Record:
    def __init__(self):
        self.models = None

    def on_stop(self, audio):
        if audio is None:
            return ("",) * 4
        if self.models is None:
            self.models = Models()
        texts = [text.strip() for text in self.models.transcribe_all(audio)]
        return *texts, texts[0]

    def on_submit(self, audio, name):
        if audio is None:
            raise gr.Error("Record audio first.")
        path = save_wav(audio, name, counted=True, directory="audio/recorded/clean")
        name = os.path.splitext(os.path.basename(path))[0]
        for noise in NOISES:
            for snr in SNRS:
                save_wav(mix(audio, snr, noise), f"{name}_{noise}_{snr}", directory="audio/recorded/noisy")
        gr.Info(f"Saved {name}.wav and {len(NOISES) * len(SNRS)} noisy files.")

    def launch(self, server_port=None):
        with gr.Blocks() as app:
            audio = gr.Audio(sources="microphone", type="numpy", label="Microphone")
            with gr.Row():
                texts = [gr.Textbox(label=f"{name} transcript", lines=3) for name in ASR_MODELS]
            name = gr.Textbox(label="Audio name")
            gr.Button("Clear").click(lambda: (None, "", "", "", ""), outputs=[audio, *texts, name])
            audio.stop_recording(self.on_stop, audio, [*texts, name])
            gr.Button("Submit", elem_id="submit").click(self.on_submit, [audio, name])
        if server_port is None:
            server_port = int(os.environ.get("PORT", 8001))
        app.launch(server_port=server_port, css="#submit{background:green!important;color:white}")

if __name__ == "__main__":
    Record().launch()
