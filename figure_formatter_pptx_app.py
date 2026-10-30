"""
Figure Formatter — PPTX export prototype
------------------------------------------
Matches the real workflow: take a figure, remove the existing text from
the image entirely (so the background stays intact but blank where
labels were), and hand back a PowerPoint slide with that cleaned image
as the background plus EDITABLE text boxes (Times New Roman, size 9)
positioned exactly where the original labels were — including rotated
axis titles (handled via figure_formatter_pptx_common.py).

Two-step review process (important for busy figures — charts with lots
of small axis numbers near plot lines/gridlines):
  1. Detect — shows every detected region as a numbered box on the
     ORIGINAL image. Nothing is erased yet. Check this against the
     original: any box sitting on top of a line you want to keep
     untouched, or any box that isn't actually text (a false positive),
     mark "no" in the Include column.
  2. Clean & Export — only regions marked "yes" get erased/replaced.
     Anything marked "no" is left completely alone in the final image.

Requires figure_formatter_app.py and figure_formatter_pptx_common.py in
the same folder.

Install:
    pip install python-pptx pandas
    (opencv-python-headless, pillow, gradio, numpy, easyocr, scikit-image
    already installed from requirements.txt)

Run:
    python figure_formatter_pptx_app.py
"""

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr

from figure_formatter_app import detect_text_regions, smart_clean_regions, draw_detected_boxes
from figure_formatter_pptx_common import (
    new_presentation_with_background,
    add_label_textbox,
    save_presentation,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run_detection(pil_image):
    """Detect only — nothing gets erased here. Lets you spot false
    positives or boxes overlapping lines before any cleaning happens."""
    if pil_image is None:
        return None, None, None
    image_np = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    regions = detect_text_regions(image_np)

    boxed_np = draw_detected_boxes(image_np, regions)
    boxed_pil = Image.fromarray(cv2.cvtColor(boxed_np, cv2.COLOR_BGR2RGB))

    rows = [[i, r["text"], r["confidence"], "yes"] for i, r in enumerate(regions)]
    df = pd.DataFrame(rows, columns=["#", "Text (edit if wrong)", "OCR confidence", "Include (yes/no)"])

    state = {"image_np": image_np, "regions": regions}
    return boxed_pil, df, state


def _read_table(table_df):
    """Returns {index: (text, include_bool)}."""
    parsed = {}
    for row in table_df.values.tolist():
        idx = int(row[0])
        text = "" if row[1] is None else str(row[1])
        include = str(row[3]).strip().lower() not in ("no", "n", "false", "0", "")
        parsed[idx] = (text, include)
    return parsed


def preview_clean(table_df, state):
    """Clean only the regions marked Include=yes, leave everything else
    untouched, so you can check the result before exporting."""
    if not state or "regions" not in state:
        return None, state

    parsed = _read_table(table_df)
    included_regions = [r for i, r in enumerate(state["regions"]) if parsed.get(i, ("", True))[1]]

    cleaned_np = smart_clean_regions(state["image_np"], included_regions)
    state["cleaned_np"] = cleaned_np
    preview = Image.fromarray(cv2.cvtColor(cleaned_np, cv2.COLOR_BGR2RGB))
    return preview, state


def build_pptx(table_df, state):
    if not state or "regions" not in state:
        return None

    parsed = _read_table(table_df)
    included_regions = [r for i, r in enumerate(state["regions"]) if parsed.get(i, ("", True))[1]]
    cleaned_np = smart_clean_regions(state["image_np"], included_regions)
    state["cleaned_np"] = cleaned_np

    prs, slide, px_to_in = new_presentation_with_background(cleaned_np)

    for i, r in enumerate(state["regions"]):
        text, include = parsed.get(i, (r["text"], True))
        if not include or not text.strip():
            continue  # excluded or blanked = leave that spot as background, no textbox
        add_label_textbox(slide, r, text, px_to_in)

    return save_presentation(prs, OUTPUT_DIR)


def build_ui():
    with gr.Blocks(title="Figure Formatter — PPTX export") as demo:
        gr.Markdown(
            "## Figure Formatter → editable PowerPoint\n"
            "1. Upload a figure, click **Detect**. Boxes are drawn on the ORIGINAL image — "
            "nothing is erased yet. Red = low OCR confidence, green = confident.\n"
            "2. In the table: fix any misread text, and set **Include** to `no` for any box that "
            "isn't real text or sits on a line/detail you want to keep untouched.\n"
            "3. Click **Preview Clean** to check the result before committing.\n"
            "4. Click **Export PPTX** for the final file — Times New Roman, size 9, editable text "
            "boxes, with rotated axis titles handled automatically."
        )
        state = gr.State(None)

        with gr.Row():
            inp = gr.Image(type="pil", label="Original figure")
            boxed_preview = gr.Image(type="pil", label="Detected regions (nothing erased yet)")

        detect_btn = gr.Button("Detect", variant="primary")
        table = gr.Dataframe(
            headers=["#", "Text (edit if wrong)", "OCR confidence", "Include (yes/no)"],
            interactive=True,
            label="Review before cleaning — set Include to 'no' to skip a region entirely",
        )

        with gr.Row():
            preview_btn = gr.Button("Preview Clean")
            export_btn = gr.Button("Export PPTX", variant="primary")

        cleaned_preview = gr.Image(type="pil", label="Cleaned background (preview)")
        out_file = gr.File(label="Download .pptx")

        detect_btn.click(fn=run_detection, inputs=inp, outputs=[boxed_preview, table, state])
        preview_btn.click(fn=preview_clean, inputs=[table, state], outputs=[cleaned_preview, state])
        export_btn.click(fn=build_pptx, inputs=[table, state], outputs=out_file)

    return demo


if __name__ == "__main__":
    build_ui().launch()
