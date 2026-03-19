# Voice Command NLP Pipeline

Offline voice command recognition using:

| Stage | Library | Purpose |
|---|---|---|
| Command detection | Edge Impulse TFLite | Classifies short audio into 9 labels |
| Speech-to-text | Vosk | Converts free-form speech to text — fully offline |
| NLP | spaCy | Extracts intent and entities from the transcription |
| Actions | action_handler.py | Placeholder hooks for your SDK / device code |

> **Fully offline** — no internet connection needed at runtime.
> The Edge Impulse model runs locally via TFLite.

---

## Project structure

```
main.py                  ← entry point
modules/
    audio_capture.py     ← microphone recording
    wake_word.py         ← Edge Impulse TFLite inference + MFCC DSP
    transcriber.py       ← Vosk offline STT
    nlp_processor.py     ← spaCy intent + entity extraction
    action_handler.py    ← action display, logging, last_result.json
.env.example             ← copy to .env and fill in paths
requirements.txt
README.md
```

Output files (written to the working directory while the pipeline runs):

| File | Contents |
|---|---|
| `actions.log` | Append-only JSON log — one entry per recognised command |
| `last_result.json` | Always contains the most recent command result |

---

## Setup

### 1 — Clone / copy the project

Place the project folder anywhere on your machine.

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

**PyAudio on Windows** — if the above fails for pyaudio, use:

```bash
pip install pipwin
pipwin install pyaudio
```

Or download the pre-built wheel from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**TFLite runtime** — if `tflite-runtime` is unavailable for your Python version:

```bash
pip install tensorflow
```

Then open `modules/wake_word.py` and comment out the `tflite-runtime` import
block — the fallback to `tensorflow` will activate automatically.

### 3 — Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4 — Download a Vosk model

Choose one from https://alphacephei.com/vosk/models :

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `vosk-model-small-en-us-0.15` | ~40 MB | Fast | Good for clear speech |
| `vosk-model-en-us-0.22` | ~1.8 GB | Slower | High accuracy |

For most use cases start with the small model.

1. Download and extract the zip.
2. Note the folder path (e.g. `C:\models\vosk-model-small-en-us-0.15`).

### 5 — Set up your .env file

```bash
copy .env.example .env
```

Edit `.env` and set:

```env
VOSK_MODEL_PATH=C:\models\vosk-model-small-en-us-0.15
TFLITE_MODEL_PATH=C:\models\human command classify\trained.tflite
```

### 6 — Edge Impulse model

The `trained.tflite` file is already in your extracted deployment folder at:

```
C:\models\human command classify\trained.tflite
```

No additional setup needed — the pipeline loads it directly.

---

## Running the pipeline

```bash
python main.py
```

You will see:

```
============================================================
   VOICE COMMAND PIPELINE
   Edge Impulse TFLite  +  Vosk STT  +  spaCy NLP
============================================================
[*] Loading Edge Impulse TFLite model...
[*] Edge Impulse model   ✓
[*] Loading Vosk STT model...
[*] Vosk model           ✓
[*] Loading spaCy NLP model...
[*] spaCy model          ✓

[*] All models ready.  Starting pipeline...
```

---

## How it works

### Listening loop

Every 2 seconds the pipeline records audio and passes it through the
Edge Impulse TFLite model.  The model classifies the audio into one of
9 labels.  Depending on the result:

```
Audio (2 s)
    │
    ▼
Edge Impulse TFLite
    │
    ├── confidence < 0.6   → discard, keep listening
    ├── "stop"             → clean shutdown
    ├── simple command     → execute_direct_command()
    │   go / left / right / up / down / yes / no
    │
    └── "hey device"       → record 5 s
                               │
                               ▼
                           Vosk STT  →  spaCy NLP  →  execute_action()
```

### Simple commands (direct path)

These are executed immediately — no full-sentence processing needed:

| Say | Action |
|---|---|
| go | Move forward |
| left | Turn left |
| right | Turn right |
| up | Move up |
| down | Move down |
| yes | Confirm |
| no | Cancel |

### Wake word + full sentence (Vosk + spaCy path)

Say **"hey device"**, wait for the *SPEAK NOW* prompt, then speak a
natural sentence:

| Example | Intent | Entities |
|---|---|---|
| "turn right at the next corner" | navigation | action: turn right |
| "go forward and stop at the door" | navigation | action: go forward |
| "what time is it" | query | — |
| "set the timer for 5 minutes" | timer | number: 5 |
| "turn on the lights" | control | action: turn on |

### Stopping the pipeline

- Say **"stop"** — detected directly by Edge Impulse
- Press **Ctrl+C**

Either method prints a session summary and exits cleanly.

---

## Integrating with your SDK

The `action_handler.py` file is the integration point for SDK engineers.
Each intent handler contains a clearly marked comment:

```python
# ── INSERT YOUR SDK CALL HERE ────────────────────────────────────────────
# e.g.  motor_sdk.navigate(action=action, target=location)
```

The `last_result.json` file is overwritten after every recognised command,
so external processes can poll it for the latest result:

```json
{
  "timestamp": "2026-03-19T10:23:52",
  "source": "edge_impulse_direct",
  "label": "left",
  "confidence": 0.94,
  "action": "Turn left"
}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Microphone not accessible` | Check mic is plugged in and Windows has permission |
| `Vosk model folder not found` | Check VOSK_MODEL_PATH in .env points to the extracted folder |
| `TFLite model not found` | Check TFLITE_MODEL_PATH in .env |
| `spaCy model not found` | Run: `python -m spacy download en_core_web_sm` |
| Low accuracy on wake word | Speak closer to the mic; reduce background noise |
| Commands not recognised | Confidence may be below 0.6 — speak clearly or retrain |
| Wrong sample rate error | The WAV must be 16 000 Hz — do not change SAMPLE_RATE |

---

## Model details

| Parameter | Value |
|---|---|
| Project | human command classification (ID 932661) |
| Labels | down, go, hey_device, left, no, right, stop, up, yes |
| Input shape | `[1, 650]` INT8 |
| MFCC | 13 coefficients, 32 filters, 256-pt FFT, 20 ms frames |
| Sample rate | 16 000 Hz |
| Confidence threshold | 0.6 |
| Inference engine | TFLite (quantised INT8) |
