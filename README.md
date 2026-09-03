# Archives AI Processing Suite — Proof of Concept

A tonight-sized build of the Queen's Archives project: local GPU inference
for audio transcription, handwriting transcription, and image captioning,
behind one small web UI.

**Total active time:** ~45-60 min, plus ~15-30 min of model downloads
happening in the background. Start the downloads in Phase 1 first, then
read the "How it all actually works" section while they finish.

---

## Phase 0 — Install Ollama (5 min)

Ollama is a **local model runner**: it downloads quantized model weights,
loads them onto your GPU, and exposes them over a local web API on
`http://localhost:11434`. It auto-detects whether you have an NVIDIA GPU —
you don't need to configure CUDA yourself for this part.

- **Windows / Mac:** download the installer from https://ollama.com and run it.
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Check it worked:
```bash
ollama --version
```

## Phase 1 — Pull the models (15-25 min, mostly waiting)

Start these now — they download in the background while you set up Python.

```bash
ollama pull gemma4:e2b     # ~7.2 GB — vision-language model (captions + handwriting)
ollama pull qwen3.5:4b     # ~3.4 GB — a second VLM to compare against gemma4
```

You only strictly need one vision model to get the proof of concept
working — pull `gemma4:e2b` first if you're tight on time, and grab
`qwen3.5:4b` later. Comparing the two on the *same* image is exactly the
"test alternative models" evaluation work the job posting mentions.

You do **not** need to `ollama pull` anything for audio — `faster-whisper`
downloads the Whisper model itself the first time you run it (Phase 3).

## Phase 2 — Python environment (5-10 min)

First install `ffmpeg` itself (a standalone program, not a pip package —
`openai-whisper` shells out to it to decode audio):

- **Windows:** `winget install ffmpeg` (or `scoop install ffmpeg` if you use Scoop)
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Then the Python side:
```bash
cd archive-ai-poc
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check whether PyTorch can see your GPU (it should auto-detect, no config
needed either way):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If that prints `False` and you *do* have an NVIDIA GPU, `pip install torch`
likely grabbed a CPU-only build. Go to https://pytorch.org/get-started/locally/,
pick your OS/CUDA version, and run the install command it generates for you
— it'll still work fine, just on CPU, if you skip this, only slower.

## Phase 3 — Test each piece from the command line (10-15 min)

Grab a test photo (any image works for captioning), a photo of some
handwriting (even your own on a piece of paper, photographed), and a short
audio clip (a voice memo is fine).

```bash
python test_pipeline.py image path/to/photo.jpg
python test_pipeline.py handwriting path/to/handwriting.jpg
python test_pipeline.py audio path/to/clip.mp3
```

**This is the actual proof of concept.** If all three print sensible
output, the hard part is done — the web UI in Phase 4 is just a thin layer
on top.

## Phase 4 — Run the web UI (5 min)

```bash
python app.py
```

Open the local URL it prints (usually `http://127.0.0.1:7860`). Pick an
operation, upload a file, hit Run.

---

## How it all actually works

**Ollama** is doing more than "running a model." It loads the quantized
weights into GPU memory once, keeps them warm, and speaks a simple HTTP API
(`/api/chat`) so any language — not just Python — can use it. The Python
`ollama` package in `ai_tools.py` is just a thin wrapper around that HTTP
call.

**Why one model does two jobs (captioning *and* handwriting).** Gemma 4 and
Qwen3.5 are *vision-language* models: an image encoder converts pixels into
the same kind of tokens the text decoder already understands, so the model
reasons over "image tokens + your text prompt" as one sequence. There's no
separate OCR step — "what's in this image" and "what does this handwriting
say" are just two different prompts against the same model. That's why
`caption_image()` and `transcribe_handwriting()` in `ai_tools.py` are
nearly identical functions with different prompt text.

**Why audio is a separate pipeline.** Whisper is an audio-in, text-out
model with a completely different architecture (encoder over spectrogram
chunks, not image patches). `openai-whisper` shells out to the standalone
`ffmpeg` program to decode audio into a waveform, then runs the Whisper
model itself on GPU via PyTorch. (There's a faster reimplementation called
`faster-whisper` that skips ffmpeg for a compiled Python extension instead
— worth trying later for speed, but it's more prone to exactly the Windows
security issue you just hit, since that extension isn't code-signed.)

**Why the code is split into three files.** `ai_tools.py` never imports
`gradio`, and `app.py` never imports `ollama` or `faster_whisper` directly.
The UI only knows "call this function, get a string back." That's what
makes the "swap to a paid API later" requirement in the JD cheap: you'd
change the *inside* of `caption_image()` to call, say, an OpenAI or Claude
API instead of Ollama, and nothing else in the project would need to know.

---

## Troubleshooting

- **`ConnectionError` / nothing at localhost:11434** — Ollama isn't
  running. It usually starts automatically after install; if not, run
  `ollama serve` in a separate terminal.
- **Out of VRAM** — switch `compute_type="float16"` to `compute_type="int8"`
  in `ai_tools.py`, or use `gemma4:e2b` instead of a larger model.
- **`ollama pull` seems stuck** — it's usually still downloading; check
  progress, it doesn't print much until it finishes a layer.
- **CUDA not found** — run `nvidia-smi` to confirm your GPU and driver are
  visible to the OS at all. `whisper.load_model("turbo")` and Ollama both
  fall back to CPU automatically if no GPU is found; things will just run
  slower.
- **`ImportError: DLL load failed ... An Application Control policy has
  blocked this file`** — this is Windows 11's Smart App Control blocking an
  unsigned compiled file (usually from the `av` package, a dependency of
  `faster-whisper`). This project uses `openai-whisper` instead specifically
  to avoid it. If you hit this from some *other* package: Settings → Privacy
  & Security → Windows Security → App & browser control → Smart App Control.
  If it's in "Evaluation" mode you can switch it off there directly. If it
  already says "On" (not evaluation), Microsoft only lets you turn it off
  through a full Windows reset — not worth doing tonight. Swapping the
  offending package for an alternative (as done here) is almost always
  faster.

## If you finish early / want to go further

- Point `caption_image()` at a real archival-style photo and see whether
  the output reads like something a catalogue would actually store — tweak
  the prompt until it does.
- Run the *same* handwriting sample through both `gemma4:e2b` and
  `qwen3.5:4b` and compare — this is literally the "testing and tuning of
  alternative models" line from the job description.
- Add a fourth function, `save_result()`, that writes the output into a
  SQLite database alongside the original filename — that's the searchable
  metadata store the JD's background section is really asking for.