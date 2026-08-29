# pyrefly: ignore [missing-import]
import fitz # PyMuPDF   
import os
from services.ocr_service import process_scanned_document
from logger import logger

def validate_text_quality(text: str):
    """Checks if the extracted text is meaningful or just noise/scanned placeholders."""
    if not text or len(text.strip()) < 80:
        return False
    
    # Check for alphabetic character density (must be > 25% letters)
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < (len(text) * 0.25):
        return False
        
    return True

def extract_digital_text(file_path: str):
    """Attempts direct text extraction from PDF metadata (Fast Path)."""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            # pyrefly: ignore [unsupported-operation]
            full_text += page.get_text()
        doc.close()
        return full_text.strip()
    except Exception as e:
        logger.warning(f"Digital extraction failed: {e}")
        return ""

def smart_extraction_pipeline(file_path: str):
    """Orchestrates Hybrid Strategy: Digital -> Validation -> OCR Fallback."""
    from time import time
    start_time = time()
    
    # 1. Image Check (Direct OCR)
    is_image = any(file_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
    if is_image:
        logger.info(f"[DocuAI] Image detected. Routing to OCR Pipeline: {os.path.basename(file_path)}")
        return process_scanned_document(file_path)

    # 2. PDF Digital Attempt
    logger.info(f"[DocuAI] Analyzing PDF for digital text: {os.path.basename(file_path)}")
    digital_text = extract_digital_text(file_path)
    
    if validate_text_quality(digital_text):
        duration = round(time() - start_time, 2)
        logger.info(f"[DocuAI] ⚡ DIGITAL extraction successful in {duration}s")
        return digital_text
    
    # 3. Fallback to OCR for Scanned PDFs
    logger.info("[DocuAI] PDF appears to be scanned or low-quality. Falling back to OCR...")
    return process_scanned_document(file_path)
