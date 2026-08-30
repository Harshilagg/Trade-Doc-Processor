# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF
import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from paddleocr import PaddleOCR
from logger import logger
from utils import ocr_cache

import threading

# Global lock for thread-safe lazy initialization
ocr_lock = threading.Lock()
_ocr_engine = None

def get_ocr_engine():
    """Lazily initializes PP-OCRv3 engine optimized for CPU speed."""
    global _ocr_engine
    if _ocr_engine is None:
        with ocr_lock:
            if _ocr_engine is None:
                logger.info("Initializing PP-OCRv3 Engine (CPU-Optimized)...")
                _ocr_engine = PaddleOCR(
                    use_angle_cls=True, 
                    lang="en",
                    ocr_version="PP-OCRv3",
                    enable_mkldnn=False,   # Disable mkldnn to avoid PIR/oneDNN errors on some CPUs
                    rec_batch_num=1,       # Lower batch for memory stability
                    cpu_threads=1          # Ensure single-thread consistency
                )
    return _ocr_engine

def preprocess_for_ocr(image_path: str, target_w: int = 800):
    """Color-preserving resize only. No grayscale, no CLAHE, no sharpening.
    PaddleOCR models are trained on color images — heavy preprocessing destroys detection.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        h, w = img.shape[:2]
        if w == target_w:
            return image_path
        
        scale = target_w / w
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        processed_path = f"proc_{target_w}_{os.path.basename(image_path)}"
        cv2.imwrite(processed_path, resized)
        
        new_h, new_w = resized.shape[:2]
        logger.info(f"[OCR] Resized {w}x{h} → {new_w}x{new_h} (color preserved)")
        return processed_path
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}. Using original.")
        return image_path

def run_ocr(engine, image_input):
    """Direct OCR call using PaddleX predict() API — handles OCRResult objects."""
    try:
        # engine.predict() returns a list of OCRResult objects (one per page)
        results = engine.predict(image_input)
        
        if not results:
            return ""
        
        all_text = []
        for page in results:
            # OCRResult behaves like a dictionary containing 'rec_texts'
            if 'rec_texts' in page:
                all_text.extend(page['rec_texts'])
        
        full_text = " ".join(all_text).strip()
        return " ".join(full_text.split())
    except Exception as e:
        logger.error(f"OCR engine error: {e}")
        return ""

def process_scanned_document(file_path: str):
    """Production OCR pipeline: single-pass, color-preserving, fast."""
    # Cache is keyed on the document's bytes, so a repeat of the same file skips
    # OCR entirely. An edited document hashes differently and misses.
    cached = ocr_cache.get(file_path)
    if cached is not None:
        return cached

    current_target = file_path
    is_pdf = file_path.lower().endswith(".pdf")
    temp_files = []
    
    # 1. Log original resolution
    orig_res = "Unknown"
    if not is_pdf:
        img = cv2.imread(file_path)
        if img is not None:
            orig_res = f"{img.shape[1]}x{img.shape[0]}"

    # 2. Convert scanned PDF → image
    if is_pdf:
        try:
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=130)
            img_path = f"temp_{os.path.basename(file_path)}.png"
            pix.save(img_path)
            current_target = img_path
            temp_files.append(img_path)
            doc.close()
            orig_res = f"{pix.width}x{pix.height}"
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise e

    try:
        engine = get_ocr_engine()
        
        # PASS 1: Fast scan at 1000px (higher precision for structured docs)
        processed = preprocess_for_ocr(current_target, target_w=1000)
        if processed != current_target:
            temp_files.append(processed)
        
        logger.info(f"[OCR] Pass 1 (1000px) starting. [Original: {orig_res}]")
        final_text = run_ocr(engine, processed)
        logger.info(f"[OCR] Pass 1 result: {len(final_text)} chars")
        
        # PASS 2: If low yield, retry at 1400px
        if len(final_text) < 100:
            logger.warning(f"[OCR] Low yield ({len(final_text)} chars). Retrying at 1400px...")
            processed_p2 = preprocess_for_ocr(current_target, target_w=1400)
            if processed_p2 != current_target:
                temp_files.append(processed_p2)
            
            retry_text = run_ocr(engine, processed_p2)
            if len(retry_text) > len(final_text):
                final_text = retry_text
                logger.info(f"[OCR] Pass 2 improved: {len(final_text)} chars")
        
        logger.info(f"[OCR] Complete. {len(final_text)} chars extracted.")

        # Store before cleanup, keyed on the original document's bytes.
        ocr_cache.put(file_path, final_text)

        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        return final_text
        
    except Exception as e:
        logger.error(f"OCR pipeline failure: {str(e)}")
        raise e
