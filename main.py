"""
main.py
-------
Entry point for the Voice Command NLP Pipeline.

Architecture
------------
  Microphone (2 s)
      │
      ▼
  Edge Impulse TFLite  ──► confidence < 0.6  ──► discard, keep listening
      │
      ├── "stop"       ──► clean shutdown
      │
      ├── simple cmd   ──► execute_direct_command()   (no Vosk needed)
      │   (go/left/right/up/down/yes/no)
      │
      └── "hey_device" ──► record 5 s  ──► Vosk STT  ──► spaCy NLP
                                                              │
                                                              ▼
                                                     execute_action()
                                                     print result dict
                                                     save last_result.json

Stop the pipeline:
  • Say  "stop"  (detected by Edge Impulse)
  • Press  Ctrl+C

Output files written after every command:
  actions.log       — append-only JSON log, one entry per line
  last_result.json  — always contains the most recent command result
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from modules.audio_capture import record_audio
from modules.wake_word import CONFIDENCE_THRESHOLD, EdgeImpulseClassifier
from modules.transcriber import load_vosk_model, transcribe_audio
from modules.nlp_processor import extract_intent_and_entities, load_spacy_model
from modules.action_handler import execute_action, execute_direct_command

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

TFLITE_MODEL_PATH: str = os.getenv(
    "TFLITE_MODEL_PATH",
    r"C:\models\human command classify\trained.tflite",
)
VOSK_MODEL_PATH: str = os.getenv("VOSK_MODEL_PATH", "")

# Labels handled as instant direct commands (no Vosk/spaCy needed)
SIMPLE_COMMANDS: set[str] = {"go", "left", "right", "up", "down", "yes", "no"}
WAKE_WORD:   str = "hey_device"
STOP_LABEL:  str = "stop"

# Recording durations
WAKE_DURATION:    float = 2.0   # seconds — for wake word / simple command
COMMAND_DURATION: float = 5.0   # seconds — full sentence after wake word


# ── Terminal formatting ────────────────────────────────────────────────────────

def _banner() -> None:
    print("\n" + "=" * 60)
    print("   VOICE COMMAND PIPELINE")
    print("   Edge Impulse TFLite  +  Vosk STT  +  spaCy NLP")
    print("=" * 60)


def _listening_prompt() -> None:
    print("\n" + "-" * 60)
    print("  Listening for wake word or simple command...")
    print("  ► Wake word  :  say  'hey device'")
    print("  ► Direct cmds:  go | left | right | up | down | yes | no")
    print("  ► Stop       :  say  'stop'  or press  Ctrl+C")
    print("-" * 60)


def _speak_prompt() -> None:
    print("\n" + "-" * 60)
    print(f"  SPEAK NOW — you have {int(COMMAND_DURATION)} seconds")
    print('  e.g. "turn right at the next corner"')
    print('       "go forward and stop at the door"')
    print('       "set the timer for 5 minutes"')
    print("-" * 60)


# ── Startup — load all models once ────────────────────────────────────────────

def _startup() -> tuple[EdgeImpulseClassifier, object]:
    """
    Load Edge Impulse, Vosk and spaCy models.
    Exits with a clear error message if any model fails to load.
    """
    _banner()

    # Edge Impulse TFLite
    print("\n[*] Loading Edge Impulse TFLite model...")
    try:
        classifier = EdgeImpulseClassifier(TFLITE_MODEL_PATH)
        print("[*] Edge Impulse model   ✓")
    except Exception as exc:
        print(f"\n[FATAL] Edge Impulse model failed: {exc}")
        sys.exit(1)

    # Vosk
    if not VOSK_MODEL_PATH:
        print(
            "\n[FATAL] VOSK_MODEL_PATH is not set in your .env file.\n"
            "        Download a model from https://alphacephei.com/vosk/models\n"
            "        and set the path in .env"
        )
        sys.exit(1)
    print("[*] Loading Vosk STT model...")
    try:
        vosk_model = load_vosk_model(VOSK_MODEL_PATH)
        print("[*] Vosk model           ✓")
    except Exception as exc:
        print(f"\n[FATAL] Vosk model failed: {exc}")
        sys.exit(1)

    # spaCy
    print("[*] Loading spaCy NLP model...")
    try:
        load_spacy_model()
        print("[*] spaCy model          ✓")
    except Exception as exc:
        print(f"\n[FATAL] spaCy model failed: {exc}")
        sys.exit(1)

    print("\n[*] All models ready.  Starting pipeline...\n")
    return classifier, vosk_model


# ── Main pipeline loop ─────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Continuous voice command loop.

    Flow per iteration:
      1.  Record WAKE_DURATION seconds.
      2.  Classify with Edge Impulse TFLite.
      3a. confidence < threshold          → skip.
      3b. label == "stop"                 → exit.
      3c. label in SIMPLE_COMMANDS        → execute_direct_command().
      3d. label == "hey_device"           → record COMMAND_DURATION seconds
                                            → Vosk STT → spaCy NLP
                                            → execute_action().
    """
    classifier, vosk_model = _startup()
    session_count: int = 0

    try:
        while True:
            _listening_prompt()

            # ── Step 1: record short audio for wake/command detection ──────
            print(f"[RECORDING] {WAKE_DURATION:.0f} s  →  speak now...")
            wake_path = record_audio(duration=WAKE_DURATION)

            if wake_path is None:
                print("[ERROR] Microphone recording failed.  Check your mic.")
                time.sleep(0.5)
                continue

            # ── Step 2: Edge Impulse classification ────────────────────────
            result = classifier.classify(wake_path)
            _unlink(wake_path)

            if result is None:
                print("[WARN] Classification returned no result.  Retrying...")
                time.sleep(0.5)
                continue

            label      = result["label"]
            confidence = result["confidence"]
            print(f"[DETECTED]  '{label}'  (confidence: {confidence:.2f})")

            # ── Step 3a: below threshold → ignore ─────────────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                print(
                    f"[SKIP] Confidence {confidence:.2f} is below "
                    f"threshold {CONFIDENCE_THRESHOLD:.2f}.  Ignoring."
                )
                time.sleep(0.5)
                continue

            # ── Step 3b: stop command → clean exit ────────────────────────
            if label == STOP_LABEL:
                print("\n[STOP]  Stop command received.  Shutting down...")
                break

            # ── Step 3c: simple command → direct execution ─────────────────
            if label in SIMPLE_COMMANDS:
                execute_direct_command(label, confidence)
                session_count += 1
                time.sleep(0.5)
                continue

            # ── Step 3d: wake word → full sentence pipeline ────────────────
            if label == WAKE_WORD:
                print(
                    "\n[PIPELINE]  Wake word detected!  "
                    "Listening for your command..."
                )
                _speak_prompt()

                print(f"[RECORDING] {COMMAND_DURATION:.0f} s  →  speak now...")
                cmd_path = record_audio(duration=COMMAND_DURATION)

                if cmd_path is None:
                    print("[ERROR] Failed to record command audio.")
                    time.sleep(0.5)
                    continue

                # Vosk STT
                print("[TRANSCRIBING]  Processing with Vosk...")
                text = transcribe_audio(cmd_path, model=vosk_model)
                _unlink(cmd_path)

                if text is None:
                    print(
                        "[WARN] Could not understand the audio.\n"
                        "       Try speaking clearly and closer to the mic."
                    )
                    time.sleep(0.5)
                    continue

                print(f'[TRANSCRIBED]  "{text}"')

                # spaCy NLP
                intent_result = extract_intent_and_entities(text)

                # Action handler
                execute_action(intent_result)

                # Pretty-print the full result dict
                print("\n[RESULT DICT]")
                print(json.dumps(
                    {**intent_result, "timestamp": datetime.now().isoformat()},
                    indent=2,
                ))

                session_count += 1

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[EXIT]  Ctrl+C received.  Shutting down...")

    finally:
        _print_summary(session_count)


def _unlink(path: str) -> None:
    """Delete a temp file silently."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _print_summary(count: int) -> None:
    print("\n" + "=" * 60)
    print("  Session complete")
    print(f"  Commands processed  :  {count}")
    print(f"  Action log          :  actions.log")
    print(f"  Last result         :  last_result.json")
    print("=" * 60)
    print("Goodbye.\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
