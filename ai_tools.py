"""
ai_tools.py
Core AI functions for the Archives multimodal processing suite.

Every function has the SAME shape: take a file path in, return a string out.
That's the "modular design" the job posting asks for. The web UI (app.py)
never talks to Ollama or Whisper directly -- it only ever calls these three
functions. That's the seam that lets you swap a local model for a paid API
later without touching the UI code at all: you'd just edit the inside of
one function.
"""

import ollama
import whisper

# ---------------------------------------------------------------------------
# Vision-language model calls: image captioning + handwriting transcription
# ---------------------------------------------------------------------------
# Both tasks below call the SAME underlying model -- only the prompt changes.
# That's the key insight for this project: a vision-language model doesn't
# have a separate "OCR mode" and "caption mode". You get different behavior
# purely by asking a different question about the same image.

VISION_MODEL = "gemma4:e2b"   # try "qwen3.5:4b" too and compare the outputs


def caption_image(image_path: str, model: str = VISION_MODEL) -> str:
    """Return a plain-language description of what an image depicts."""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Describe this image in 2-3 sentences, in the style of an "
                "archival catalogue description. Mention notable people, "
                "objects, setting, and anything that looks historically "
                "significant."
            ),
            "images": [image_path],
        }],
    )
    return response["message"]["content"]


def transcribe_handwriting(image_path: str, model: str = VISION_MODEL) -> str:
    """Return a machine-readable transcript of handwritten text in an image."""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Transcribe the handwritten text in this image exactly as "
                "written, preserving original spelling and punctuation. If "
                "a word is illegible, write [illegible] in its place. "
                "Output ONLY the transcription, with no commentary before "
                "or after it."
            ),
            "images": [image_path],
        }],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Audio transcription -- a separate pipeline (Whisper, not the VLM above)
# ---------------------------------------------------------------------------

_whisper_model = None  # loaded once and reused -- loading it is the slow part


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # device=None lets PyTorch auto-pick: GPU if torch.cuda.is_available(),
        # otherwise CPU. No manual device string needed.
        _whisper_model = whisper.load_model("turbo")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Return a full text transcript of an audio file."""
    model = _get_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()