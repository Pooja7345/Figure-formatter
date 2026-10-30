"""
Figure Formatter — local chat app (no API, no cost)
-------------------------------------------------------
Same chat-style interface as figure_formatter_chat_app.py, but instead
of an LLM understanding free-form text, this uses simple keyword
commands you type — everything runs 100% locally, no external API,
no internet needed after the one-time model download, no per-use cost.

Cleaning is deferred until 'preview' or 'export' — 'detect' only shows
numbered boxes on the ORIGINAL image so you can spot false positives or
boxes overlapping lines/details you want to keep, and 'clear' them
before anything actually gets erased. Rotated axis titles are handled
automatically on export.

Requires figure_formatter_app.py and figure_formatter_pptx_common.py in
the same folder.

Commands (type these into the chat box):
    detect                     -> scan the image, show detected boxes (nothing erased yet)
    list                       -> show all detected labels and their status
    set 3 to Pressure (kPa)    -> override label #3's text before export
    clear 3                    -> exclude label #3 entirely (left untouched, no textbox)
    preview                    -> clean the background (using current clear/set choices) and show it
    reset                      -> discard edits, start over from the original image
    export                     -> build the .pptx with current labels/overrides
    help                       -> show this command list

Install: nothing new — same requirements.txt as before.

Run:
    python figure_formatter_chat_local.py
"""

import os
import re

import cv2
import numpy as np
from PIL import Image
import gradio as gr

from figure_formatter_app import detect_text_regions, smart_clean_regions, draw_detected_boxes
from figure_formatter_pptx_common import (
    new_presentation_with_background,
    add_label_textbox,
    save_presentation,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

HELP_TEXT = (
    "Commands:\n"
    "  detect                    - scan the image, show boxes (nothing erased yet)\n"
    "  list                      - show detected labels\n"
    "  set 3 to Pressure (kPa)   - override label #3's text\n"
    "  clear 3                   - exclude label #3 (left untouched, no textbox)\n"
    "  preview                   - clean the background and show the result\n"
    "  reset                     - discard edits, start over\n"
    "  export                    - build the .pptx\n"
    "  help                      - show this list"
)

SET_PATTERN = re.compile(r"^\s*set\s+(\d+)\s+(?:to|=|:)\s*(.+?)\s*$", re.IGNORECASE)
CLEAR_PATTERN = re.compile(r"^\s*clear\s+(\d+)\s*$", re.IGNORECASE)


def list_labels(state):
    if not state.get("regions"):
        return "No detection run yet — type 'detect' first."
    lines = []
    for i, r in enumerate(state["regions"]):
        if i in state.get("cleared", set()):
            status = "EXCLUDED (left untouched, no textbox)"
        else:
            text = state.get("overrides", {}).get(i, r["text"])
            conf_flag = " [low confidence]" if r["flagged"] else ""
            status = f'"{text}"{conf_flag}'
        lines.append(f"  #{i}: {status}")
    return "Detected labels:\n" + "\n".join(lines)


def included_regions(state):
    cleared = state.get("cleared", set())
    return [r for i, r in enumerate(state["regions"]) if i not in cleared]


def build_pptx(state):
    cleaned_np = smart_clean_regions(state["image_np"], included_regions(state))
    state["cleaned_np"] = cleaned_np

    prs, slide, px_to_in = new_presentation_with_background(cleaned_np)

    overrides = state.get("overrides", {})
    cleared = state.get("cleared", set())

    for i, r in enumerate(state["regions"]):
        if i in cleared:
            continue
        text = overrides.get(i, r["text"])
        if not text.strip():
            continue
        add_label_textbox(slide, r, text, px_to_in)

    return save_presentation(prs, OUTPUT_DIR)


def handle_command(message, state):
    """Returns (reply_text, updated_state, preview_pil_or_None, file_path_or_None)."""
    if state.get("image_np") is None:
        return "Upload a figure first, then type 'detect'.", state, None, None

    text = message.strip()
    lower = text.lower()

    if lower in ("help", "commands", "?"):
        return HELP_TEXT, state, None, None

    if lower in ("detect", "scan"):
        regions = detect_text_regions(state["image_np"])
        state["regions"] = regions
        state["overrides"] = {}
        state["cleared"] = set()
        state.pop("cleaned_np", None)
        flagged = sum(1 for r in regions if r["flagged"])
        boxed = draw_detected_boxes(state["image_np"], regions)
        preview = Image.fromarray(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
        reply = (
            f"Detected {len(regions)} label(s), {flagged} low-confidence. Nothing erased yet.\n"
            f"Red box = low confidence, green = confident. Use 'clear #' to exclude any box "
            f"that isn't real text or overlaps a line you want to keep.\n\n" + list_labels(state)
        )
        return reply, state, preview, None

    if lower in ("list", "show", "labels"):
        return list_labels(state), state, None, None

    if lower in ("preview", "clean"):
        if not state.get("regions"):
            return "No detection run yet — type 'detect' first.", state, None, None
        cleaned_np = smart_clean_regions(state["image_np"], included_regions(state))
        state["cleaned_np"] = cleaned_np
        preview = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_BGR2RGB))
        return "Cleaned preview (only included labels were touched).", state, preview, None

    if lower in ("reset", "start over", "undo all"):
        state["regions"] = []
        state["overrides"] = {}
        state["cleared"] = set()
        state.pop("cleaned_np", None)
        return "Reset. Type 'detect' to start again.", state, None, None

    m = SET_PATTERN.match(text)
    if m:
        idx, new_text = int(m.group(1)), m.group(2)
        if not state.get("regions"):
            return "No detection run yet — type 'detect' first.", state, None, None
        if idx < 0 or idx >= len(state["regions"]):
            return f"No label #{idx}. Type 'list' to see valid indices.", state, None, None
        state.setdefault("overrides", {})[idx] = new_text
        state.setdefault("cleared", set()).discard(idx)
        return f'Label #{idx} set to "{new_text}".', state, None, None

    m = CLEAR_PATTERN.match(text)
    if m:
        idx = int(m.group(1))
        if not state.get("regions"):
            return "No detection run yet — type 'detect' first.", state, None, None
        if idx < 0 or idx >= len(state["regions"]):
            return f"No label #{idx}. Type 'list' to see valid indices.", state, None, None
        state.setdefault("cleared", set()).add(idx)
        return f"Label #{idx} excluded — will be left untouched, no textbox added.", state, None, None

    if lower in ("export", "save", "pptx", "download"):
        if not state.get("regions"):
            return "No detection run yet — type 'detect' first.", state, None, None
        out_path = build_pptx(state)
        return f"Exported: {os.path.basename(out_path)}", state, None, out_path

    return f"Didn't recognize that command. {HELP_TEXT}", state, None, None


def on_upload(pil_image):
    if pil_image is None:
        return {}, None
    image_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    state = {"image_np": image_np, "regions": [], "overrides": {}, "cleared": set()}
    return state, pil_image


def chat_turn(message, history, state):
    reply, state, preview, file_path = handle_command(message, state)
    history = history + [(message, reply)]
    return history, state, preview, file_path


def build_ui():
    with gr.Blocks(title="Figure Formatter — local chat") as demo:
        gr.Markdown(
            "## Figure Formatter (local chat, no API)\n"
            "Upload a figure, then type commands: `detect`, `list`, "
            "`set 2 to Temperature (K)`, `clear 4`, `preview`, `export`, `reset`, `help`.\n\n"
            "`detect` only shows boxes — nothing is erased until `preview` or `export`."
        )
        state = gr.State({})
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload figure")
            preview = gr.Image(type="pil", label="Detected boxes / cleaned preview")
        chatbot = gr.Chatbot(label="Commands", height=300)
        msg = gr.Textbox(label="Type a command and press Enter")
        out_file = gr.File(label="Download .pptx (after 'export')")

        inp.change(fn=on_upload, inputs=inp, outputs=[state, preview])
        msg.submit(
            fn=chat_turn, inputs=[msg, chatbot, state], outputs=[chatbot, state, preview, out_file]
        ).then(lambda: "", None, msg)

    return demo


if __name__ == "__main__":
    build_ui().launch()
