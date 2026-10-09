# Noise Cancellation

This repository compares automatic speech recognition on short English queries from the news and ecommerce domains. It uses open-source SLURP data together with clean speech mixed with background noise at fixed signal-to-noise ratios, and scores both open-source models and commercial speech-to-text APIs. The comparison evaluates how well each solution handles background noise using WER, CER metrics.

## Data

Two sources are used.

- Recorded speech is captured with `src/app/gradio_record.py`. On submit, the recorded clean clip is stored under `audio/recorded/clean`. The same clip is then mixed with three noise clips — claps and cheers, traffic and wind, and retail ambient noise — at SNR levels of −5, 0, 5, and 10 dB. Those files are stored under `audio/recorded/noisy`. A lower SNR corresponds to louder noise. Mixing is implemented in `src/utils/audio_utils.py`.
- Open-source speech comes from the SLURP test set (`qmeeus/slurp`). News queries (`news_query`) and ecommerce queries (`lists_createoradd`, `lists_query`, `lists_remove`, `takeaway_order`, and `takeaway_query`) are used. Clips are cached as wav files under `audio/open_source`, with an index in `audio/open_source/slurp.csv`.

## Models

Three open-source models are loaded locally with Hugging Face `transformers` and cached under `models/`:


| Name          | Checkpoint                                 |
| ------------- | ------------------------------------------ |
| whisper-small | `distil-whisper/distil-small.en`           |
| whisper-large | `openai/whisper-large-v3`                  |
| parakeet      | `ai-and-i-project/parakeet-tdt-0.6b-v2-hf` |


Four commercial APIs are also evaluated. Each service requires an API key in `.env`.

- Deepgram Nova-3
- ElevenLabs Scribe v2
- AssemblyAI Universal-3.5 Pro
- Speechmatics Enhanced

## Evaluation

`audio_transcription.ipynb` transcribes every clip. Reference and hypothesis text are normalized with OpenAI Whisper’s English text normalizer, then scored with `jiwer`. The reported metrics are word error rate (WER), character error rate (CER), and transcription latency. Cost is estimated from published batch rates in US dollars per hour of audio.

## Results

Median word error rate, character error rate, and latency across groups are taken from `results/overall_metrics.csv`. Cost is the published batch rate in US dollars per hour of audio.


| Model         | Source      | Median WER (%) | Median CER (%) | Median latency (ms) | Cost ($/hr) |
| ------------- | ----------- | -------------- | -------------- | ------------------- | ----------- |
| whisper-small | open-source | 20.8           | 12.2           | 83.0                | 0.0         |
| whisper-large | open-source | 8.1            | 5.5            | 621.5               | 0.0         |
| parakeet      | open-source | 14.1           | 7.6            | 39.5                | 0.0         |
| deepgram      | paid        | 14.0           | 9.0            | 1750.0              | 0.258       |
| elevenlabs    | paid        | 7.2            | 4.9            | 1510.0              | 0.22        |
| assemblyai    | paid        | 6.5            | 3.4            | 1394.5              | 0.15        |
| speechmatics  | paid        | 8.4            | 4.5            | 2370.0              | 0.23        |


## Findings

### Open-source

- Whisper-large is the most accurate of the three local models and the closest to the commercial APIs.
- For low-latency applications, Parakeet is a suitable choice. It is the fastest model and more accurate than Whisper-small.

### Commercial

- AssemblyAI has the lowest median error rates and the lowest batch rate among the paid services ($0.15/hr).

## Setup

Python 3.12 is assumed. Install the pinned dependencies, then provide the API keys above before running the notebook or either application.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The following Gradio interface records speech from the microphone. It mixes each recording with the three noise types at four SNR levels and saves the clean clip together with the noisy mixes. 

```bash
python -m src.app.gradio_record
```

The demonstration interface accepts microphone or uploaded audio, transcribes it, and mixes a selected noise type while searching for the optimal SNR at which the noisy transcript remains close to the clean one.

```bash
python -m src.app.gradio_demo
```

