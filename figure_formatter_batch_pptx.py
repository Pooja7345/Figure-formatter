"""
Figure Formatter — batch PPTX export (fully local, no external APIs)
-----------------------------------------------------------------------
Point this at a FOLDER of figures instead of one image at a time.
For each image it runs the same clean-background + detect-text pipeline
as figure_formatter_pptx_app.py, and writes one .pptx per image into an
output folder — Times New Roman, size 9, editable text boxes, exact
label positions, no padding, rotated axis titles handled automatically.

Since batch mode has no per-image review step, EVERY detected region
gets cleaned automatically — including any false positives (non-text
shapes OCR mistook for text) or labels sitting close to plot
lines/gridlines. For busy charts (lots of small axis numbers near
colored lines, dense gridlines) this can occasionally blur into a
nearby line. Use this for simpler figures/diagrams with few labels; for
dense charts, use figure_formatter_pptx_app.py instead, which lets you
review and exclude risky regions before anything is erased.

Requires figure_formatter_app.py and figure_formatter_pptx_common.py in
the same folder.

Install (same as before, nothing new):
    pip install -r requirements.txt

Run:
    python figure_formatter_batch_pptx.py "C:\\path\\to\\your\\figures\\folder"

    If you omit the path, it looks for a folder named "input" next to
    this script.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

from figure_formatter_app import detect_text_regions, smart_clean_regions
from figure_formatter_pptx_common import (
    new_presentation_with_background,
    add_label_textbox,
    save_presentation,
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def build_pptx_for_image(image_path, output_dir):
    pil_image = Image.open(image_path).convert("RGB")
    image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    regions = detect_text_regions(image_np)
    cleaned_np = smart_clean_regions(image_np, regions)

    prs, slide, px_to_in = new_presentation_with_background(cleaned_np)

    for r in regions:
        text = r["text"]
        if not text.strip():
            continue
        add_label_textbox(slide, r, text, px_to_in)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = save_presentation(prs, output_dir, base_name=base_name)

    low_conf = sum(1 for r in regions if r["flagged"])
    return out_path, len(regions), low_conf


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIR

    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
        print("Create an 'input' folder next to this script and put your figures in it,")
        print('or run: python figure_formatter_batch_pptx.py "C:\\path\\to\\your\\figures"')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [
        f for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    if not images:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(images)} image(s). Processing...\n")

    for i, filename in enumerate(images, 1):
        image_path = os.path.join(input_dir, filename)
        print(f"[{i}/{len(images)}] {filename} ... ", end="", flush=True)
        try:
            out_path, n_labels, n_low_conf = build_pptx_for_image(image_path, OUTPUT_DIR)
            flag = f" ({n_low_conf} low-confidence — review these)" if n_low_conf else ""
            print(f"done -> {os.path.basename(out_path)} [{n_labels} labels{flag}]")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nAll done. Output .pptx files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
