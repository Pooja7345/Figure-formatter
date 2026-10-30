"""
Figure Formatter — chat-driven prototype
------------------------------------------
Same pipeline as figure_formatter_app.py, but instead of one "Process"
button, you type instructions in a chat box (e.g. "clean up the blurry
text on the graph and sharpen it more", "redo detection, it's missing
the y-axis label", "upscale it 3x") and an LLM (Claude) decides which
pipeline function(s) to call.

Requires figure_formatter_app.py in the SAME folder — this file imports
the pipeline functions from it instead of duplicating them.

Setup:
    pip install anthropic
    (all the packages from requirements.txt, if not already installed)

    Set your API key as an environment variable before running:
      Windows (Command Prompt):  set ANTHROPIC_API_KEY=your-key-here
      Windows (PowerShell):      $env:ANTHROPIC_API_KEY="your-key-here"
      Mac/Linux:                 export ANTHROPIC_API_KEY=your-key-here

    Get a key at https://console.anthropic.com

Run:
    python figure_formatter_chat_app.py
"""

import json

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from anthropic import Anthropic

from figure_formatter_app import (
    detect_text_regions,
    build_mask,
    inpaint_flagged_regions,
    redraw_text,
    sharpen_and_upscale,
)

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You control an image-editing pipeline for preparing
figures for journal submission. You have tools to detect text, clean
(remove + redraw in Times New Roman) specific text regions, upscale/
sharpen the image, and reset to the original. The user will describe
what they want in plain language — pick the right tool call(s) to
satisfy it. Always call detect_text first if no detection has been run
yet in this conversation. Keep your reply to the user short and
concrete about what you actually did."""

TOOLS = [
    {
        "name": "detect_text",
        "description": "Run OCR text detection on the current image. Use first, or to redo detection with different sensitivity (e.g. user says it's missing something, or over-flagging things that are fine).",
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence_threshold": {"type": "number", "description": "0-1, lower = flags more text as bad. Default 0.55."},
                "min_box_height": {"type": "number", "description": "pixels, text shorter than this gets flagged. Default 10."},
            },
        },
    },
    {
        "name": "clean_regions",
        "description": "Remove and redraw text regions in Times New Roman. Call detect_text first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Which detected region indices to clean. Omit or pass an empty list to clean everything currently flagged.",
                }
            },
        },
    },
    {
        "name": "upscale_image",
        "description": "Sharpen and upscale the current image.",
        "input_schema": {
            "type": "object",
            "properties": {"scale": {"type": "number", "description": "e.g. 1.5, 2, 3. Default 2."}},
        },
    },
    {
        "name": "reset_image",
        "description": "Discard all edits, revert to the originally uploaded image and clear detected regions.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name, tool_input, state):
    """Execute one tool call against `state`, return a short result string."""
    if name == "detect_text":
        conf = tool_input.get("confidence_threshold", 0.55)
        min_h = tool_input.get("min_box_height", 10)
        # temporarily patch module-level thresholds used by detect_text_regions
        import figure_formatter_app as ffa
        ffa.CONFIDENCE_THRESHOLD = conf
        ffa.MIN_BOX_HEIGHT_PX = min_h
        state["regions"] = detect_text_regions(state["current"])
        flagged = sum(1 for r in state["regions"] if r["flagged"])
        return f"Detected {len(state['regions'])} text region(s), {flagged} flagged as low quality."

    if name == "clean_regions":
        if not state["regions"]:
            return "No detection has been run yet — call detect_text first."
        indices = tool_input.get("indices") or []
        regions = state["regions"]
        if indices:
            for r in regions:
                r["_temp_flag"] = r["flagged"]
                r["flagged"] = False
            for i in indices:
                if 0 <= i < len(regions):
                    regions[i]["flagged"] = True
        mask = build_mask(state["current"].shape, regions)
        cleaned = inpaint_flagged_regions(state["current"], mask)
        state["current"] = redraw_text(cleaned, regions)
        if indices:
            for r in regions:
                if "_temp_flag" in r:
                    r["flagged"] = r.pop("_temp_flag")
        return f"Cleaned {len(indices) if indices else sum(1 for r in regions if r['flagged'])} region(s)."

    if name == "upscale_image":
        scale = tool_input.get("scale", 2)
        state["current"] = sharpen_and_upscale(state["current"], scale=scale)
        return f"Upscaled/sharpened by {scale}x."

    if name == "reset_image":
        state["current"] = state["original"].copy()
        state["regions"] = []
        return "Reverted to the original image."

    return f"Unknown tool: {name}"


def chat_turn(message, history, state):
    if state.get("original") is None:
        return history + [(message, "Upload a figure first, then tell me what to do with it.")], state, None

    messages = state.get("claude_messages", [])
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )

    # Execute any tool calls, feed results back, get the final text reply
    while response.stop_reason == "tool_use":
        tool_results = []
        assistant_content = [b.model_dump() for b in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        for block in response.content:
            if block.type == "tool_use":
                result_text = run_tool(block.name, block.input, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

    final_text = "".join(b.text for b in response.content if b.type == "text") or "Done."
    messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
    state["claude_messages"] = messages

    out_pil = Image.fromarray(cv2.cvtColor(state["current"], cv2.COLOR_BGR2RGB))
    return history + [(message, final_text)], state, out_pil


def on_upload(pil_image):
    if pil_image is None:
        return {}, None
    image_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    state = {"original": image_np.copy(), "current": image_np.copy(), "regions": [], "claude_messages": []}
    return state, pil_image


def build_ui():
    with gr.Blocks(title="Figure Formatter — chat") as demo:
        gr.Markdown(
            "## Figure Formatter (chat-driven)\n"
            "Upload a figure, then tell it what to do — "
            "e.g. *\"clean up the blurry axis labels\"*, "
            "*\"redo detection, it's being too aggressive\"*, "
            "*\"upscale it 3x\"*, *\"start over\"*."
        )
        state = gr.State({})
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload figure")
            out = gr.Image(type="pil", label="Current result")
        chatbot = gr.Chatbot(label="Instructions", height=300)
        msg = gr.Textbox(label="Type an instruction and press Enter")

        inp.change(fn=on_upload, inputs=inp, outputs=[state, out])
        msg.submit(fn=chat_turn, inputs=[msg, chatbot, state], outputs=[chatbot, state, out]).then(
            lambda: "", None, msg
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
