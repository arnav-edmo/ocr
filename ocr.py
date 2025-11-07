#!/usr/bin/env python3
"""
ocr_pipeline.py
Starter pipeline:
pdf -> images -> preprocess -> layout detection -> OCR ensemble (TrOCR + Tesseract) ->
table detection placeholder -> semantic extraction -> save PAGE-XML + canonical JSON -> index to Elasticsearch.

NOTES:
- This is a starter: replace model names/weights & paths as required.
- Use on-prem hosting for models for defence data.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import cv2
import numpy as np
import layoutparser as lp
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from elasticsearch import Elasticsearch   # uncomment if ES indexing needed
# import cascade_tabnet or other table model as required

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- CONFIG ----
PDF_POPPLER_PATH = "/usr/bin"  # change if needed
TESSERACT_CMD = "/usr/bin/tesseract"  # ensure installed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# TrOCR (smaller/light models recommended for on-prem)
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"  # replace if necessary

# Layout model from LayoutParser (Detectron2 backbone)
LAYOUT_MODEL = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"   # good starting model (object classes) - train on your data.

OUTPUT_DIR = "output"
ES_INDEX = "defence_documents"  # optional

# ---- Utilities ----
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def pdf_to_images(pdf_path, dpi=300):
    logger.info("Converting PDF to images: %s", pdf_path)
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=PDF_POPPLER_PATH)
    page_files = []
    ensure_dir(os.path.join(OUTPUT_DIR, "pages"))
    for i, img in enumerate(images, start=1):
        fname = os.path.join(OUTPUT_DIR, "pages", f"page_{i:04d}.png")
        img.save(fname)
        page_files.append(fname)
    return page_files

def preprocess_image(image_path):
    # basic deskew and contrast; optionally integrate Real-ESRGAN for small text
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # simple threshold & denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # simple deskew using minAreaRect on text contours (approx)
    coords = np.column_stack(np.where(gray < 250))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # write back preprocessed
    pre_path = image_path.replace(".png", "_pre.png")
    cv2.imwrite(pre_path, img)
    return pre_path

# ---- Layout detection (layoutparser) ----
def detect_layout(image_path):
    logger.info("Detecting layout on %s", image_path)
    model = lp.Detectron2LayoutModel(
        config_path=LAYOUT_MODEL,
        label_map={0:'Text',1:'Title',2:'List',3:'Table',4:'Figure'}
    )
    image = Image.open(image_path).convert("RGB")
    layout = model.detect(image)
    # Convert to serializable format
    blocks = []
    for i, b in enumerate(layout):
        blocks.append({
            "block_id": f"blk_{i}",
            "type": b.type.lower(),
            "bbox": [int(b.block.x_1), int(b.block.y_1), int(b.block.get_width()), int(b.block.get_height())],
            "confidence": float(b.score) if hasattr(b, "score") else None
        })
    return blocks, image

# ---- OCR ensemble for a block (TrOCR + Tesseract fallback) ----
class OCREnsemble:
    def __init__(self, trocr_model_name=TROCR_MODEL_NAME, device="cpu"):
        self.device = device
        # load TrOCR processor + model
        try:
            self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name).to(device)
            logger.info("Loaded TrOCR model %s", trocr_model_name)
        except Exception as e:
            logger.warning("TrOCR load failed: %s. Falling back to Tesseract-only.", e)
            self.processor = None
            self.model = None

    def ocr_trocr(self, pil_img):
        if self.model is None:
            return None, 0.0
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # no confidence returned by TrOCR here; set a placeholder
        return preds, 0.9

    def ocr_tesseract(self, pil_img):
        txt = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')
        # get mean confidence from tesseract hOCR? (This is a placeholder)
        return txt.strip(), 0.85

    def recognize(self, pil_img):
        # try TrOCR first, then Tesseract; choose by heuristic
        trocr_txt, trocr_conf = self.ocr_trocr(pil_img)
        tess_txt, tess_conf = self.ocr_tesseract(pil_img)
        # heuristic: prefer longer, non-empty result with higher confidence
        chosen = None
        conf = 0.0
        if trocr_txt and len(trocr_txt) > 0:
            chosen, conf = trocr_txt, trocr_conf
        else:
            chosen, conf = tess_txt, tess_conf
        # simple ensemble: if Tesseract substantially differs and Tess_conf > trocr_conf, use Tesseract
        if tess_txt and tess_conf > conf + 0.05:
            chosen, conf = tess_txt, tess_conf
        return chosen, conf

# ---- Simple semantic parser (regex-based) ----
import re
def semantic_parse_block_text(text):
    kvs = []
    # displacement
    m = re.search(r"[Dd]isplacement[^\d]*([0-9,]+)\s*(?:tonnes|t|tons)?\s*(.*)", text)
    if m:
        raw = m.group(0)
        val = int(m.group(1).replace(",", ""))
        qual = m.group(2).strip()
        kvs.append({"field_name":"Displacement_tonnes", "raw_text": raw, "value_normalized": val, "units":"tonnes", "qualifier":qual, "confidence":0.9})
    # dimensions
    m2 = re.search(r"[Dd]imensions[^\d]*([\d\.]+)\s*[x×]\s*([\d\.]+)\s*[x×]\s*([\d\.]+)", text)
    if m2:
        dims = [float(m2.group(i)) for i in range(1,4)]
        kvs.append({"field_name":"Dimensions_metres", "raw_text": m2.group(0), "value_normalized": dims, "units":"metres", "confidence":0.9})
    # speed
    m3 = re.search(r"[Ss]peed[^\d]*([0-9]+)\s*knots", text)
    if m3:
        kvs.append({"field_name":"Speed_knots", "raw_text": m3.group(0), "value_normalized": int(m3.group(1)), "units":"knots", "confidence":0.9})
    return kvs

# ---- save canonical JSON & simple PAGE-like JSON ----
def build_document_json(pdf_path, pages_output):
    doc_json = {
        "document_id": f"doc_{uuid.uuid4().hex[:8]}",
        "source_file": os.path.basename(pdf_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": pages_output,
        "indexing": {"full_text": " ".join([p.get("full_text","") for p in pages_output])},
        "audit": {"pipeline_version":"starter_v0.1", "created_by":"ocr_pipeline.py", "created_at": datetime.utcnow().isoformat()+"Z", "human_reviewed": False}
    }
    return doc_json

# ---- main processing per PDF ----
def process_pdf(pdf_path):
    ensure_dir(OUTPUT_DIR)
    page_files = pdf_to_images(pdf_path)
    trocr = OCREnsemble()
    pages_output = []
    for i, page_img in enumerate(page_files, start=1):
        pre = preprocess_image(page_img)
        blocks, pil_image = detect_layout(pre)
        page_entry = {"page_no": i, "width": pil_image.width, "height": pil_image.height, "image": pre, "layout_blocks": [], "key_values": [], "entities": []}
        full_text_parts = []
        for b in blocks:
            x,y,w,h = b["bbox"]
            crop = pil_image.crop((x,y,x+w,y+h))
            text, conf = trocr.recognize(crop)
            kvs = semantic_parse_block_text(text or "")
            # simplified entity detection: if uppercase words > threshold, mark as ship name
            ent_conf = 0.0
            ents = []
            for tok in (text or "").split():
                if len(tok) >= 3 and tok.isupper():
                    ents.append({"type":"ship_name","text":tok,"bbox":[x,y,w,h],"confidence":0.85})
                    ent_conf = max(ent_conf, 0.85)
            block_obj = {
                "block_id": b["block_id"],
                "type": b["type"],
                "bbox": b["bbox"],
                "text": text,
                "ocr_confidence": conf,
                "confidence": b.get("confidence", None)
            }
            page_entry["layout_blocks"].append(block_obj)
            page_entry["key_values"].extend(kvs)
            page_entry["entities"].extend(ents)
            if text:
                full_text_parts.append(text)
        page_entry["full_text"] = "\n".join(full_text_parts)
        page_entry["page_confidence"] = float(np.mean([ (lb.get("ocr_confidence") or 0.8) for lb in page_entry["layout_blocks"] ]))
        pages_output.append(page_entry)
    # build and save doc json
    doc = build_document_json(pdf_path, pages_output)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(pdf_path) + ".json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    logger.info("Saved canonical JSON to %s", out_path)
    return out_path

# ---- CLI ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="input PDF file")
    args = parser.parse_args()
    result = process_pdf(args.pdf)
    print("Result JSON:", result)
