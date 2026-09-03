
# Archives AI Processing Suite — Proof of Concept

## What this is

A working local-GPU pipeline for three tasks archivists need: transcribing
audio, transcribing handwritten documents, and generating captions for
historical images — all running entirely offline on local hardware, with
no data ever leaving the machine.

Built as a proof of concept for Queen's University Archives' proposed AI
workstation project. It uses the exact models named in that project's
technical brief (Ollama-served vision-language models for image/handwriting
work, Whisper for audio) and implements the two requirements called out
explicitly: a modular backend where any model can be swapped for a paid API
later with no changes elsewhere in the code, and comparative testing across
models rather than committing to a single one.

**Stack:** Python, Ollama (local model serving), Gemma 4 E2B & Qwen3.5-4B
(vision-language models), OpenAI Whisper turbo (audio), Gradio (web UI).

**A few real problems solved along the way :**
- Gemma 4 E2B hallucinated badly on open-ended photo captioning at default
  settings, opted to use Qwen 3.5-4B instead.
- Handwriting transcription hit a degenerate repetition loop (the model
  getting stuck outputting the same token); fixed with a repeat penalty
  and a separate, task-specific temperature setting from captioning.
- Hit a Windows 11 Smart App Control block on a compiled audio-decoding
  dependency; resolved by swapping to a subprocess-based alternative
  instead of disabling OS security features.

---

## Screenshots

**gemma4:e2b Model image caption and handwriting transcription**
<img width="1917" height="1137" alt="Screenshot 2026-09-03 162134" src="https://github.com/user-attachments/assets/150b3804-8c54-4185-9e14-398f5cbbfa62" />

<img width="1917" height="1136" alt="Screenshot 2026-09-03 164612" src="https://github.com/user-attachments/assets/3c3d3f2e-bdae-4824-b461-e4ff323f3042" />



**qwen3.5:4b  Model image caption and handwriting transcription**
<img width="1916" height="1135" alt="Screenshot 2026-09-03 164343" src="https://github.com/user-attachments/assets/7f753904-7149-4a73-a5c9-2ac13302e689" />

<img width="1917" height="1078" alt="Screenshot 2026-09-03 165027" src="https://github.com/user-attachments/assets/82352420-c111-41dc-939b-c458ebff2a6d" />

**Audio transcription**
<img width="1917" height="1095" alt="Screenshot 2026-09-03 165315" src="https://github.com/user-attachments/assets/29cd45f3-3cff-4afd-a9ae-3fd9dd783f99" />

---

---

## Setup 


**Total active time:** It takes about ~45-60 min, plus ~15-30 min of model downloads
happening in the background. 

If you wanna set this up for your self start the downloads in Phase 1 first, then
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

```bash
ollama pull gemma4:e2b     # ~7.2 GB — vision-language model (captions + handwriting)
ollama pull qwen3.5:4b     # ~3.4 GB — a second VLM to compare against gemma4
```

You do **not** need to `ollama pull` anything for audio — `openai-whisper`
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

Grab a test photo , a photo of some
handwriting , and a short
audio clip.

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

Open the local URL it prints. Pick an
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

## Troubleshooting (these are some issues i came across)

- **`ConnectionError` / nothing at localhost:11434** — Ollama isn't
  running. It usually starts automatically after install; if not, run
  `ollama serve` in a separate terminal.
- **Out of VRAM** — in `ai_tools.py`, switch `whisper.load_model("turbo")`
  to a smaller size like `"small"` or `"base"`, and prefer `gemma4:e2b`
  over larger vision models.
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


---

## Conclusion

This proof of concept shows that all three tasks in the project brief —
audio transcription, handwritten document transcription, and image
captioning — run end-to-end on local, GPU-based inference, with no data
leaving the machine and no per-file API cost. It also surfaced findings
that matter for the fuller build: small local models can be reliable at
one task and unreliable at another (Gemma 4 E2B handled captioning
reasonably but looped on handwriting until tuned), generation settings
like temperature and repeat penalty need to be chosen per task rather than
set once globally, and comparing models side-by-side on the same input is
necessary rather than optional.

This is a proof of concept, not a finished tool. A production build would
add batch processing, a persistent searchable metadata store, broader
evaluation across more model candidates on real archival material, and a
more polished interface. The modular design here — one function per task,
with the web UI completely decoupled from the model-calling code — means
those are extensions to what exists, not a rewrite of it.

## Acknowledgments

Built by Natan Atnafu. Developed with AI-assisted pair programming using Claude
(Anthropic) — for initial code scaffolding, debugging platform-specific
issues (Windows Smart App Control blocking a compiled dependency, PATH
configuration problems), and guidance on model-tuning decisions like
temperature and repeat penalty. All setup, hands-on debugging, model
testing, and evaluation were carried out personally, on local hardware,
end to end.
