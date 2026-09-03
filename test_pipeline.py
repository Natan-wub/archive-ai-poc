"""
test_pipeline.py
Quick sanity check for each function WITHOUT the web UI.
Run this first -- it's faster to debug in a terminal than in a browser.

Usage:
    python test_pipeline.py image path/to/photo.jpg [model]
    python test_pipeline.py handwriting path/to/scan.jpg [model]
    python test_pipeline.py audio path/to/clip.mp3

The optional [model] lets you compare vision models on the same file, e.g.:
    python test_pipeline.py handwriting "C:\\path\\scan.jpg" qwen3.5:4b
"""

import sys
from ai_tools import caption_image, transcribe_handwriting, transcribe_audio

MODES = {
    "image": caption_image,
    "handwriting": transcribe_handwriting,
    "audio": transcribe_audio,
}

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in MODES:
        print(__doc__)
        sys.exit(1)

    mode, path = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) == 4 else None

    print(f"Running '{mode}' on {path}" + (f" with model={model}" if model else "") + " ...\n")

    fn = MODES[mode]
    result = fn(path, model=model) if (model and mode != "audio") else fn(path)

    print("--- OUTPUT ---")
    print(result)