
"""
Figure Formatter — sample prototype

Features:
1. Detect text using EasyOCR.
2. Flag low-confidence and small text regions.
3. Remove flagged text using OpenCV.
4. Smart background reconstruction.
5. Redraw text using Pillow.
6. Sharpen and upscale the image.
7. Display before/after results in Gradio.
"""

import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# OCR READER
# --------------------------------------------------

_READER = None


def get_reader():
    global _READER

    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False)

    return _READER


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.55
MIN_BOX_HEIGHT_PX = 10

INPAINT_PADDING = 3
INPAINT_RADIUS = 5

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf",
    "times.ttf",
]


# --------------------------------------------------
# FONT LOADING
# --------------------------------------------------

def load_font(size: int):

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


# --------------------------------------------------
# TEXT DETECTION
# --------------------------------------------------

def detect_text_regions(image_np: np.ndarray):
    """Detect text using EasyOCR and return region information."""

    reader = get_reader()
    results = reader.readtext(image_np)

    regions = []

    for box, text, conf in results:

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))

        height = y1 - y0

        flagged = (
            conf < CONFIDENCE_THRESHOLD
            or height < MIN_BOX_HEIGHT_PX
        )

        regions.append({
            "box": (x0, y0, x1, y1),
            "text": text,
            "confidence": round(float(conf), 3),
            "height": height,
            "flagged": flagged,
        })

    return regions


# --------------------------------------------------
# DETECTED BOX VISUALIZATION
# --------------------------------------------------

def draw_detected_boxes(image_np, regions):
    """Show detected regions on the original image."""

    out = image_np.copy()

    for i, r in enumerate(regions):

        x0, y0, x1, y1 = r["box"]

        color = (
            (0, 0, 255) if r["flagged"]
            else (0, 180, 0)
        )

        cv2.rectangle(
            out,
            (x0, y0),
            (x1, y1),
            color,
            2
        )

        cv2.putText(
            out,
            str(i),
            (x0, max(y0 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return out


# --------------------------------------------------
# MASK CREATION
# --------------------------------------------------

def build_mask(image_shape, regions):
    """Create a mask over flagged text regions only."""

    mask = np.zeros(
        image_shape[:2],
        dtype=np.uint8
    )

    for r in regions:

        if not r["flagged"]:
            continue

        x0, y0, x1, y1 = r["box"]
        p = INPAINT_PADDING

        cv2.rectangle(
            mask,
            (max(x0 - p, 0), max(y0 - p, 0)),
            (
                min(x1 + p, image_shape[1]),
                min(y1 + p, image_shape[0])
            ),
            255,
            thickness=-1
        )

    return mask


# --------------------------------------------------
# ORIGINAL OPENCV INPAINTING
# --------------------------------------------------

def inpaint_flagged_regions(image_np, mask):

    return cv2.inpaint(
        image_np,
        mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )


# --------------------------------------------------
# SMART BACKGROUND RECONSTRUCTION
# --------------------------------------------------

def _dominant_color_fill(
    ring_pixels,
    tolerance=20,
    majority_frac=0.55
):
    """Find the dominant surrounding color."""

    flat = ring_pixels.reshape(
        -1,
        ring_pixels.shape[-1]
    ).astype(np.float32)

    quantized = flat // 8 * 8

    colors, counts = np.unique(
        quantized,
        axis=0,
        return_counts=True
    )

    dominant = colors[np.argmax(counts)]

    dist = np.linalg.norm(
        flat - dominant,
        axis=-1
    )

    close = flat[dist < tolerance]

    frac_close = len(close) / len(flat)

    if frac_close >= majority_frac:
        return np.median(close, axis=0)

    return None


def smart_clean_regions(
    image_np,
    regions,
    padding=3,
    ring=10
):
    """
    Remove selected text regions while preserving
    surrounding backgrounds and gradients.
    """

    out = image_np.copy()

    h, w = image_np.shape[:2]

    reconstruct_mask = np.zeros(
        (h, w),
        dtype=bool
    )

    for r in regions:

        x0, y0, x1, y1 = r["box"]

        bx0 = max(x0 - padding, 0)
        by0 = max(y0 - padding, 0)

        bx1 = min(x1 + padding, w)
        by1 = min(y1 + padding, h)

        rx0 = max(bx0 - ring, 0)
        ry0 = max(by0 - ring, 0)

        rx1 = min(bx1 + ring, w)
        ry1 = min(by1 + ring, h)

        ring_mask = np.ones(
            (ry1 - ry0, rx1 - rx0),
            dtype=bool
        )

        iy0 = by0 - ry0
        iy1 = by1 - ry0

        ix0 = bx0 - rx0
        ix1 = bx1 - rx0

        ring_mask[iy0:iy1, ix0:ix1] = False

        ring_pixels = image_np[
            ry0:ry1,
            rx0:rx1
        ][ring_mask]

        if ring_pixels.size == 0:

            reconstruct_mask[
                by0:by1,
                bx0:bx1
            ] = True

            continue

        fill_color = _dominant_color_fill(ring_pixels)

        if fill_color is not None:

            out[
                by0:by1,
                bx0:bx1
            ] = fill_color

        else:

            reconstruct_mask[
                by0:by1,
                bx0:bx1
            ] = True

    if reconstruct_mask.any():

        from skimage.restoration import inpaint_biharmonic

        try:
            reconstructed = inpaint_biharmonic(
                out,
                reconstruct_mask,
                channel_axis=-1
            )

        except TypeError:
            reconstructed = inpaint_biharmonic(
                out,
                reconstruct_mask,
                multichannel=True
            )

        out = np.clip(
            reconstructed * 255
            if reconstructed.max() <= 1.0
            else reconstructed,
            0,
            255
        ).astype(np.uint8)

    return out


# --------------------------------------------------
# REDRAW TEXT
# --------------------------------------------------

def redraw_text(image_np, regions, overrides=None):

    overrides = overrides or {}

    pil_img = Image.fromarray(
        cv2.cvtColor(
            image_np,
            cv2.COLOR_BGR2RGB
        )
    )

    draw = ImageDraw.Draw(pil_img)

    for i, r in enumerate(regions):

        if not r["flagged"]:
            continue

        text = overrides.get(i, r["text"])

        x0, y0, x1, y1 = r["box"]

        box_h = max(y1 - y0, 8)

        font = load_font(
            size=int(box_h * 0.9)
        )

        draw.text(
            (x0, y0),
            text,
            fill=(0, 0, 0),
            font=font
        )

    result = np.array(pil_img)

    return cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )


# --------------------------------------------------
# SHARPEN AND UPSCALE
# --------------------------------------------------

def sharpen_and_upscale(image_np, scale=2.0):

    h, w = image_np.shape[:2]

    upscaled = cv2.resize(
        image_np,
        (
            int(w * scale),
            int(h * scale)
        ),
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(
        upscaled,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        upscaled,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened


# --------------------------------------------------
# COMPLETE PROCESSING PIPELINE
# --------------------------------------------------

def run_pipeline(image_np):

    regions = detect_text_regions(image_np)

    mask = build_mask(
        image_np.shape,
        regions
    )

    cleaned = inpaint_flagged_regions(
        image_np,
        mask
    )

    redrawn = redraw_text(
        cleaned,
        regions
    )

    final_image = sharpen_and_upscale(
        redrawn
    )

    return final_image, regions


# --------------------------------------------------
# GRADIO PROCESS FUNCTION
# --------------------------------------------------

def process(pil_image):

    if pil_image is None:
        return None, "Upload a figure first."

    image_np = cv2.cvtColor(
        np.array(pil_image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

    final_np, regions = run_pipeline(image_np)

    final_pil = Image.fromarray(
        cv2.cvtColor(
            final_np,
            cv2.COLOR_BGR2RGB
        )
    )

    report_lines = [
        f"Detected {len(regions)} text region(s):"
    ]

    for r in regions:

        status = (
            "FLAGGED → removed & redrawn"
            if r["flagged"]
            else "kept as-is"
        )

        report_lines.append(
            f"'{r['text']}' "
            f"(confidence={r['confidence']}, "
            f"height={r['height']}px) — {status}"
        )

    report = "\n".join(report_lines)

    return final_pil, report


# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------

def build_ui():

    import gradio as gr

    with gr.Blocks(
        title="Figure Formatter — prototype"
    ) as demo:

        gr.Markdown(
            textwrap.dedent("""
            ## Figure Formatter

            Upload a figure. Low-confidence or tiny text
            gets removed and redrawn. The image is then
            sharpened and upscaled.

            Review the output before using it in a
            real publication.
            """)
        )

        with gr.Row():

            inp = gr.Image(
                type="pil",
                label="Original figure"
            )

            out = gr.Image(
                type="pil",
                label="Processed figure"
            )

        report = gr.Textbox(
            label="Detection report",
            lines=10
        )

        btn = gr.Button(
            "Process",
            variant="primary"
        )

        btn.click(
            fn=process,
            inputs=inp,
            outputs=[out, report]
        )

    return demo


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    print("OpenCV module:", cv2.__file__)
    print("OpenCV version:", cv2.__version__)
    print("cvtColor available:", hasattr(cv2, "cvtColor"))

    build_ui().launch()
