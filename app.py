"""
app.py
Minimal web UI for the Archives multimodal processing suite.

Built with Gradio because it gets a working file-upload UI running in about
20 lines -- good for tonight's proof of concept. The job posting's "simple
web UI" requirement doesn't specify a framework, so this is a legitimate
answer to it. If you later want something closer to a production app (or
to match your Flask experience), you'd swap this file for a Flask app --
ai_tools.py wouldn't need to change at all. That's the whole point of
keeping the AI logic in its own module.
"""

import gradio as gr
from ai_tools import caption_image, transcribe_handwriting, transcribe_audio


def process(operation: str, model: str, file):
    if file is None:
        return "Upload a file first."
    path = file.name if hasattr(file, "name") else file

    if operation == "Caption image":
        return caption_image(path, model=model)
    if operation == "Transcribe handwriting":
        return transcribe_handwriting(path, model=model)
    if operation == "Transcribe audio":
        return transcribe_audio(path)  # audio doesn't use the vision model picker
    return "Unknown operation."


with gr.Blocks(title="Archives AI Processing Suite") as demo:
    gr.Markdown("# Archives AI Processing Suite (proof of concept)")
    operation = gr.Radio(
        ["Caption image", "Transcribe handwriting", "Transcribe audio"],
        label="Operation",
        value="Caption image",
    )
    model = gr.Radio(
        ["gemma4:e2b", "qwen3.5:4b"],
        label="Vision model (ignored for audio)",
        value="gemma4:e2b",
    )
    file_input = gr.File(label="Upload a file")
    output = gr.Textbox(label="Output", lines=8)
    run_button = gr.Button("Run")
    run_button.click(fn=process, inputs=[operation, model, file_input], outputs=output)

if __name__ == "__main__":
    demo.launch()