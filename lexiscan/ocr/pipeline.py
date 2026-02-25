"""
PDF → Pixels → Text OCR Pipeline
----------------------------------
Dependencies:
    pip install pdf2image pytesseract Pillow
    System: Install Poppler
        macOS:   brew install poppler
        Ubuntu:  sudo apt install poppler-utils
        Windows: https://github.com/oschwartz10612/poppler-windows/releases
"""

import gc
import logging
from pathlib import Path

import os
import sys

# Set environment variables BEFORE importing pytesseract
if os.name == 'nt':
    # Windows specific paths - can be overridden by environment variables
    if 'TESSDATA_PREFIX' not in os.environ:
        os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

    # Add Tesseract and Poppler to PATH if likely locations exist
    tesseract_path = r'C:\Program Files\Tesseract-OCR'
    # Check for Poppler in common locations or environment variable
    poppler_path = os.environ.get('POPPLER_PATH')

    new_path = os.environ['PATH']
    if os.path.exists(tesseract_path) and tesseract_path not in new_path:
        new_path = tesseract_path + os.pathsep + new_path
    if poppler_path and os.path.exists(poppler_path) and poppler_path not in new_path:
        new_path = poppler_path + os.pathsep + new_path

    os.environ['PATH'] = new_path

import pytesseract

if os.name == 'nt':
    # Only set explicit command path on Windows if not already set
    tesseract_exe = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_exe):
        pytesseract.pytesseract.pytesseract_cmd = tesseract_exe

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

# Logging configuration
LOG_LEVEL = logging.DEBUG  # Set to logging.INFO to reduce verbosity
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

DPI = 300          # Sweet spot: enough detail for OCR, not too large for RAM
GRAYSCALE = True   # Cuts memory ~3x vs RGB; OCR doesn't need colour
LANG = "eng"       # Tesseract language pack(s), e.g. "eng+fra" for multi-language


# ── Core helpers ──────────────────────────────────────────────────────────────

def get_page_count(pdf_path: str) -> int:
    """Return total page count without loading the whole PDF into RAM."""
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(pdf_path)
    return info["Pages"]


def convert_page(pdf_path: str, page_number: int):
    """
    Convert a single PDF page to a PIL Image.

    page_number is 1-indexed (Poppler convention).
    Returns a PIL Image in grayscale (or RGB if GRAYSCALE=False).
    """
    images = convert_from_path(
        pdf_path,
        dpi=DPI,
        first_page=page_number,
        last_page=page_number,   # Load exactly one page — key to memory safety
        grayscale=GRAYSCALE,
    )
    return images[0]  # Always a list; we asked for one page so grab index 0


def ocr_image(image) -> str:
    """Run Tesseract OCR on a PIL Image and return the extracted text."""
    return pytesseract.image_to_string(image, lang=LANG)


def ocr_image_spatial(image) -> pd.DataFrame:
    """
    Extract text WITH spatial information using Tesseract.
    
    Returns a DataFrame containing:
    - text: The word recognized
    - left, top: X, Y coordinates of the word's bounding box
    - width, height: Dimensions of the bounding box
    - conf: Confidence score (0-100) for that word
    - page, block, par, line, word: Hierarchy indices
    
    This is crucial for:
    1. Finding where each word is on the page (layout reconstruction)
    2. Filtering low-confidence predictions (< 50% = probably wrong)
    3. Applying custom logic (e.g., ignore words in margins)
    """
    # Extract all OCR data with spatial information
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=LANG)
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(ocr_data)
    
    # Convert string columns to numeric where appropriate
    numeric_cols = ['page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter out non-text rows (empty text or confidence = -1)
    df = df[(df['text'].str.strip() != '') & (df['conf'] >= 0)].reset_index(drop=True)
    
    log.debug(f"Spatial OCR: Found {len(df)} words with confidence >= 0%")
    
    return df


def filter_by_confidence(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """
    Filter spatial OCR results by confidence threshold.
    
    Args:
        df: DataFrame from ocr_image_spatial()
        threshold: Minimum confidence score (0-100)
        
    Returns:
        Filtered DataFrame with only high-confidence words
    """
    initial_count = len(df)
    df_filtered = df[df['conf'] >= threshold].reset_index(drop=True)
    removed = initial_count - len(df_filtered)
    
    if removed > 0:
        log.info(f"Confidence filter: Removed {removed} low-confidence words (threshold: {threshold}%)")
    
    return df_filtered


def reconstruct_text_from_spatial(df: pd.DataFrame) -> str:
    """
    Reconstruct paragraph-like text from spatial DataFrame.
    
    Logic:
    - Words on the same line stay together
    - New line = new line
    - Preserves layout structure
    """
    text_lines = []
    
    # Group by line number to reconstruct lines
    for line_num, group in df.groupby('line_num', sort=False):
        # Sort by left position to maintain word order (left-to-right)
        line_words = group.sort_values('left')['text'].tolist()
        line_text = ' '.join(line_words)
        text_lines.append(line_text)
    
    # Join lines with newlines to preserve structure
    reconstructed_text = '\n'.join(text_lines)
    
    return reconstructed_text


def visualize_spatial_ocr(image_pil, df: pd.DataFrame, output_path: str | None = None):
    """
    Draw bounding boxes around detected words on the image.
    
    Args:
        image_pil: PIL Image
        df: DataFrame from ocr_image_spatial()
        output_path: If provided, saves the annotated image here
        
    Useful for:
    - Debugging OCR failures
    - Seeing which regions have low confidence
    - Understanding layout detection
    """
    # Convert PIL to OpenCV
    image_cv = pil_to_cv2(image_pil)
    
    # Define color based on confidence thresholds
    def get_color(conf):
        if conf >= 80:
            return (0, 255, 0)  # Green: high confidence
        elif conf >= 50:
            return (0, 165, 255)  # Orange: medium confidence
        else:
            return (0, 0, 255)  # Red: low confidence
    
    # Draw bounding box and text for each word
    for idx, row in df.iterrows():
        x = int(row['left'])
        y = int(row['top'])
        w = int(row['width'])
        h = int(row['height'])
        text = row['text']
        conf = int(row['conf'])
        
        color = get_color(conf)
        
        # Draw rectangle
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 2)
        
        # Put confidence score above the box
        label = f"{conf}%"
        cv2.putText(image_cv, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert back to PIL
    annotated_pil = cv2_to_pil(image_cv)
    
    # Save if requested
    if output_path:
        annotated_pil.save(output_path)
        log.info(f"Annotated image saved to: {output_path}")
    
    return annotated_pil


# ── OpenCV Pre-processing Pipeline ────────────────────────────────────────────

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV (BGR) format."""
    # Convert PIL to numpy array (RGB)
    rgb_array = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_image


def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) back to PIL Image."""
    if len(cv2_image.shape) == 2:  # grayscale, no channel conversion needed
        return Image.fromarray(cv2_image)
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def grayscale_image(cv2_img):
    """Convert image to grayscale."""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    log.debug("✓ Grayscaling complete")
    return gray


def otsu_binarization(gray_img):
    """
    Apply Otsu's thresholding (automatic threshold calculation).
    
    Otsu's method analyzes the histogram to find the optimal threshold value
    that separates foreground (text) from background (paper) automatically.
    Returns a binary image (0 or 255 only).
    """
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("✓ Otsu binarization complete")
    return binary


def remove_noise(binary_img):
    """
    Remove salt-and-pepper noise using morphological transformations.
    
    Strategy:
    1. Erosion: Removes small white specks (noise)
    2. Dilation: Removes small black specks and restores text thickness
    This combination is called "Closing" - it fills small holes in the foreground.
    """
    # Create a morphological kernel (3x3 rectangle)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Morphological Closing: Dilation followed by Erosion
    # Fills small holes in the text/foreground
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Morphological Opening: Erosion followed by Dilation
    # Removes small noise specks from the background
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    denoised = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    log.debug("✓ Noise removal complete")
    return denoised


def deskew_image(binary_img):
    """
    Correct tilted/skewed images by finding the rotation angle.
    
    Algorithm:
    1. Find all black pixels (contours)
    2. Calculate the minimum area rectangle that bounds all black pixels
    3. Extract the rotation angle from that rectangle
    4. Rotate the image by the negative of that angle
    """
    # Find contours of text regions
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 5:
        log.debug("⚠ Too few contours; skipping deskew")
        return binary_img
    
    # Merge all contours into one
    all_points = np.vstack(contours)
    
    # Get the minimum area rectangle around all text
    rect = cv2.minAreaRect(all_points)
    angle = rect[2]  # Rotation angle
    
    # Angle adjustment: OpenCV returns angles in range [-90, 0]
    # We want small rotations, so if angle < -45, adjust it
    if angle < -45:
        angle = angle + 90
    
    # Skip deskew if angle is very small (already straight)
    if abs(angle) < 0.5:
        log.debug("✓ Image already straight (angle: {:.2f}°)".format(angle))
        return binary_img
    
    log.debug(f"✓ Deskewing: rotating by {angle:.2f}°")
    
    # Get image dimensions
    h, w = binary_img.shape
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Apply rotation (use white background for rotated areas)
    deskewed = cv2.warpAffine(
        binary_img,
        rotation_matrix,
        (w, h),
        borderValue=255  # White background
    )
    
    return deskewed


def preprocess_image(pil_image):
    """
    Complete preprocessing pipeline: converts PIL → OpenCV → applies all steps → returns PIL.
    
    Steps:
    1. Convert to OpenCV format (BGR)
    2. Grayscale
    3. Otsu's Binarization
    4. Noise Removal (Morphological ops)
    5. Deskewing
    6. Convert back to PIL for OCR
    """
    log.info("Starting image preprocessing pipeline...")
    
    # 1. PIL → OpenCV
    cv2_img = pil_to_cv2(pil_image)
    
    # 2. Grayscale
    gray_img = grayscale_image(cv2_img)
    
    # 3. Otsu's Binarization (threshold)
    binary_img = otsu_binarization(gray_img)
    
    # 4. Noise Removal
    denoised = remove_noise(binary_img)
    
    # 5. Deske wing
    deskewed = deskew_image(denoised)
    
    # 6. Convert back to PIL
    preprocessed_pil = cv2_to_pil(deskewed)
    
    log.info("✓ Preprocessing complete!")
    return preprocessed_pil


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, output_path: str | None = None) -> dict[int, str]:
    """
    Process a PDF page by page, returning a dict of {page_number: text}.

    Optionally writes a plain-text file to output_path.
    """
    pdf_path = str(pdf_path)
    results: dict[int, str] = {}

    try:
        total_pages = get_page_count(pdf_path)
    except (PDFInfoNotInstalledError, PDFPageCountError) as e:
        log.error("Could not read PDF — is Poppler installed? (%s)", e)
        raise

    log.info("Starting OCR: %s (%d pages, %d DPI)", pdf_path, total_pages, DPI)

    for page_num in range(1, total_pages + 1):
        log.info("  Processing page %d / %d …", page_num, total_pages)

        # 1. Rasterise this one page only
        image = convert_page(pdf_path, page_num)

        # 2. Preprocess the image (grayscale, binarization, noise removal, deskewing)
        preprocessed_image = preprocess_image(image)

        # 3. OCR the preprocessed pixels
        text = ocr_image(preprocessed_image)

        # 4. Store the result
        results[page_num] = text

        # 5. Explicitly release images from RAM before the next iteration
        del image, preprocessed_image
        gc.collect()

    log.info("OCR complete. %d pages processed.", total_pages)

    # ── Optional: write full text to a file ───────────────────────────────────
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for page_num, text in results.items():
                f.write(f"{'='*60}\nPage {page_num}\n{'='*60}\n")
                f.write(text)
                f.write("\n\n")
        log.info("Text saved to: %s", output_path)

    return results


# ── Advanced variant: adaptive DPI ────────────────────────────────────────────

def detect_small_print(image, threshold_px: int = 20) -> bool:
    """
    Heuristic: if most text characters are shorter than threshold_px,
    the document likely has small print and benefits from higher DPI.

    Uses Tesseract's layout analysis (no extra dependencies).
    """
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        lang=LANG,
    )
    heights = [h for h in data["height"] if h > 0]
    if not heights:
        return False
    avg_height = sum(heights) / len(heights)
    return avg_height < threshold_px


def process_pdf_adaptive_dpi(pdf_path: str) -> dict[int, str]:
    """
    Like process_pdf(), but bumps DPI to 400 for pages with small print.
    Costs a bit more time but improves footnote / fine-print accuracy.
    """
    pdf_path = str(pdf_path)
    results: dict[int, str] = {}
    total_pages = get_page_count(pdf_path)

    for page_num in range(1, total_pages + 1):
        # First pass at 300 DPI to check character size
        image = convert_page(pdf_path, page_num)

        if detect_small_print(image):
            log.info("  Page %d: small print detected, re-rendering at 400 DPI", page_num)
            del image
            gc.collect()
            # Re-render at higher resolution
            images = convert_from_path(
                pdf_path, dpi=400,
                first_page=page_num, last_page=page_num,
                grayscale=GRAYSCALE,
            )
            image = images[0]
        else:
            log.info("  Page %d: standard DPI OK", page_num)

        # Preprocess and OCR
        preprocessed_image = preprocess_image(image)
        results[page_num] = ocr_image(preprocessed_image)
        del image, preprocessed_image
        gc.collect()

    return results


# ── Zone-Based Extraction (Intelligent Field Detection) ──────────────────────

class DocumentZone:
    """Represents a region on a page for targeted extraction."""
    
    def __init__(self, name: str, x_min: int, x_max: int, y_min: int, y_max: int):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def contains(self, word_x: int, word_y: int) -> bool:
        """Check if a word at (word_x, word_y) is within this zone."""
        return (self.x_min <= word_x <= self.x_max and 
                self.y_min <= word_y <= self.y_max)
    
    def __repr__(self):
        return f"Zone({self.name}, x=[{self.x_min}-{self.x_max}], y=[{self.y_min}-{self.y_max}])"


def create_standard_zones(page_width: int = 2000, page_height: int = 2600) -> dict:
    """
    Create standard document zones for typical A4/Letter sized pages at 300 DPI.
    
    Assumes:
    - Page width: ~2000 pixels (8.5" × 300 DPI)
    - Page height: ~2600 pixels (11" × 300 DPI)
    
    Returns a dict of DocumentZone objects for common regions.
    """
    zones = {
        'header': DocumentZone('Header', 0, page_width, 0, int(page_height * 0.08)),
        'title': DocumentZone('Title', 0, page_width, int(page_height * 0.08), int(page_height * 0.20)),
        'top_right': DocumentZone('Top Right', int(page_width * 0.6), page_width, 0, int(page_height * 0.15)),
        'body': DocumentZone('Body', 0, page_width, int(page_height * 0.20), int(page_height * 0.85)),
        'footer': DocumentZone('Footer', 0, page_width, int(page_height * 0.85), page_height),
        'left_margin': DocumentZone('Left Margin', 0, int(page_width * 0.10), 0, page_height),
        'right_margin': DocumentZone('Right Margin', int(page_width * 0.90), page_width, 0, page_height),
    }
    return zones


def extract_zone(df: pd.DataFrame, zone: DocumentZone) -> pd.DataFrame:
    """
    Extract all words within a specific zone.
    
    Args:
        df: DataFrame with 'left' and 'top' columns (from ocr_image_spatial)
        zone: DocumentZone object defining the region
    
    Returns:
        Filtered DataFrame containing only words in the zone
    """
    zone_data = df[
        (df['left'] >= zone.x_min) & (df['left'] <= zone.x_max) &
        (df['top'] >= zone.y_min) & (df['top'] <= zone.y_max)
    ].copy()
    
    log.debug(f"Zone '{zone.name}': Found {len(zone_data)} words")
    return zone_data


def extract_all_zones(df: pd.DataFrame, zones: dict) -> dict[str, pd.DataFrame]:
    """
    Extract words for all zones at once.
    
    Args:
        df: Spatial OCR DataFrame
        zones: Dict of zone_name -> DocumentZone
    
    Returns:
        Dict of zone_name -> DataFrame with words in that zone
    """
    results = {}
    for zone_name, zone in zones.items():
        results[zone_name] = extract_zone(df, zone)
    return results


def get_zone_text(df: pd.DataFrame, zone: DocumentZone, min_confidence: int = 50) -> str:
    """
    Get reconstructed text from a specific zone.
    
    Args:
        df: Spatial OCR DataFrame
        zone: DocumentZone to extract from
        min_confidence: Only include words with confidence >= this
    
    Returns:
        Text from the zone, reconstructing line structure
    """
    zone_data = extract_zone(df, zone)
    high_conf = filter_by_confidence(zone_data, threshold=min_confidence)
    
    if len(high_conf) == 0:
        return ""
    
    text = reconstruct_text_from_spatial(high_conf)
    return text


def analyze_document_structure(df: pd.DataFrame, page_width: int = 2000, page_height: int = 2600) -> dict:
    """
    Analyze document structure to identify likely zones.
    
    Looks for:
    - Dense regions (lots of text clustered together)
    - Sparse regions (few words = margins/whitespace)
    - Text concentration percentages per zone
    
    Returns a report of document structure.
    """
    zones = create_standard_zones(page_width, page_height)
    zone_stats = {}
    
    for zone_name, zone in zones.items():
        zone_data = extract_zone(df, zone)
        if len(zone_data) > 0:
            avg_conf = zone_data['conf'].mean()
            zone_stats[zone_name] = {
                'word_count': len(zone_data),
                'avg_confidence': avg_conf,
                'words': ' '.join(zone_data['text'].head(10).tolist()),  # First 10 words
            }
        else:
            zone_stats[zone_name] = {
                'word_count': 0,
                'avg_confidence': 0,
                'words': '(empty)',
            }
    
    return zone_stats


# ── HOCR Output (HTML-based OCR with layout preservation) ────────────────────

def generate_hocr(df: pd.DataFrame, page_num: int, page_width: int = 2000, page_height: int = 2600) -> str:
    """
    Generate hOCR (HTML OCR) output.
    
    hOCR is an open standard for encoding OCR data as HTML.
    It preserves:
    - Word location (bbox)
    - Confidence score (x_size attribute)
    - Hierarchy (carea, par, line, ocrx_word)
    
    Example output:
    <span class='ocrx_word' id='word_1_1' title='bbox 850 154 1068 197; x_size 43; x_conf 85'>
        AYUSH
    </span>
    
    Benefits:
    - Can be opened in browsers
    - Locates text on original PDF when clicked
    - Professional legal software standard
    - Easy to parse with standard XML tools
    
    Returns:
        HTML string with hOCR markup
    """
    html_parts = []
    
    # Header with page info
    html_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html xmlns="http://www.w3.org/1999/xhtml">')
    html_parts.append('<head>')
    html_parts.append(f'<meta charset="utf-8" />')
    html_parts.append(f'<title>hOCR - Page {page_num}</title>')
    html_parts.append(f'<meta name="ocr-system" content="Tesseract + PDF Pipeline" />')
    html_parts.append(f'<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />')
    html_parts.append('</head>')
    html_parts.append(f'<body>')
    
    # Page container
    html_parts.append(f'<div class="ocr_page" id="page_{page_num}" title="bbox 0 0 {page_width} {page_height}">')
    
    # Group words by carea (text block), paragraph, line
    current_block = None
    current_para = None
    current_line = None
    
    block_div = None
    para_div = None
    line_div = None
    
    for idx, row in df.iterrows():
        block = row.get('block_num', 1)
        para = row.get('par_num', 1)
        line = row.get('line_num', 1)
        
        # Start new block if needed
        if block != current_block:
            if line_div:
                line_div = line_div.rstrip() + '</span>'
                html_parts.append(line_div)
            if para_div:
                para_div = para_div.rstrip() + '</div>'
                html_parts.append(para_div)
            if block_div:
                block_div = block_div.rstrip() + '</div>'
                html_parts.append(block_div)
            
            block_div = f'<div class="ocr_carea" id="carea_{page_num}_{block}">'
            html_parts.append(block_div)
            current_block = block
        
        # Start new paragraph if needed
        if para != current_para:
            if line_div:
                line_div = line_div.rstrip() + '</span>'
                html_parts.append(line_div)
            if para_div:
                para_div = para_div.rstrip() + '</div>'
                html_parts.append(para_div)
            
            para_div = f'<p class="ocr_par" id="par_{page_num}_{block}_{para}">'
            html_parts.append(para_div)
            current_para = para
        
        # Start new line if needed
        if line != current_line:
            if line_div:
                line_div = line_div.rstrip() + '</span>'
                html_parts.append(line_div)
            
            # Estimate line bbox from typical character height
            line_bbox = f"{int(row['left'])} {int(row['top'])} {int(row['left'] + row['width'])} {int(row['top'] + row['height'] * 1.2)}"
            line_div = f'<span class="ocr_line" id="line_{page_num}_{block}_{para}_{line}" title="bbox {line_bbox}">'
            html_parts.append(line_div)
            current_line = line
        
        # Add word
        word_id = f"word_{page_num}_{block}_{para}_{line}_{row.get('word_num', idx)}"
        bbox = f"{int(row['left'])} {int(row['top'])} {int(row['left'] + row['width'])} {int(row['top'] + row['height'])}"
        conf = int(row['conf'])
        word_height = int(row['height'])
        word_text = row['text']
        
        # x_size = font size approximation, x_conf = confidence
        word_html = f'<span class="ocrx_word" id="{word_id}" title="bbox {bbox}; x_size {word_height}; x_conf {conf}">{word_text}</span> '
        html_parts.append(word_html)
    
    # Close all open tags
    if line_div:
        html_parts.append('</span>')
    if para_div:
        html_parts.append('</div>')
    if block_div:
        html_parts.append('</div>')
    
    # Close page and document
    html_parts.append('</div>')  # Close ocr_page
    html_parts.append('</body>')
    html_parts.append('</html>')
    
    return '\n'.join(html_parts)


def save_hocr(df_dict: dict[int, pd.DataFrame], output_path: str | None = None):
    """
    Save all pages as hOCR HTML files.
    
    Args:
        df_dict: Dictionary of page_num -> DataFrame (from process_pdf_spatial)
        output_path: Base path for output (default: pdf_name_hocr/)
    
    Creates one HTML file per page: page_1.hocr, page_2.hocr, etc.
    """
    if output_path is None:
        output_path = "hocr_output"
    
    import os
    os.makedirs(output_path, exist_ok=True)
    
    for page_num, df in df_dict.items():
        hocr_content = generate_hocr(df, page_num)
        hocr_file = os.path.join(output_path, f"page_{page_num}.hocr")
        
        with open(hocr_file, 'w', encoding='utf-8') as f:
            f.write(hocr_content)
        
        log.info(f"Saved hOCR: {hocr_file} ({len(df)} words)")


# ── Spatial OCR Variant: Extract text WITH location data ──────────────────────

def process_pdf_spatial(pdf_path: str, confidence_threshold: int = 50) -> dict[int, pd.DataFrame]:
    """
    Process PDF with SPATIAL OCR extraction.
    
    Returns a dict of {page_number: DataFrame} where each DataFrame contains:
    - text: Recognized word
    - left, top, width, height: Bounding box coordinates
    - conf: Confidence score (0-100)
    
    Args:
        pdf_path: Path to PDF file
        confidence_threshold: Only include words with confidence >= this value (0-100)
    
    Returns:
        Dictionary mapping page numbers to DataFrames with spatial data
    
    Usage:
        spatial_results = process_pdf_spatial("document.pdf", confidence_threshold=60)
        for page_num, df in spatial_results.items():
            print(f"Page {page_num}: {len(df)} words detected")
            high_conf = df[df['conf'] >= 80]
            print(f"  High confidence words: {len(high_conf)}")
    """
    pdf_path = str(pdf_path)
    results: dict[int, pd.DataFrame] = {}

    try:
        total_pages = get_page_count(pdf_path)
    except (PDFInfoNotInstalledError, PDFPageCountError) as e:
        log.error("Could not read PDF — is Poppler installed? (%s)", e)
        raise

    log.info("Starting SPATIAL OCR: %s (%d pages, %d DPI)", pdf_path, total_pages, DPI)

    for page_num in range(1, total_pages + 1):
        log.info("  Processing page %d / %d (spatial) …", page_num, total_pages)

        # 1. Rasterise
        image = convert_page(pdf_path, page_num)

        # 2. Preprocess
        preprocessed_image = preprocess_image(image)

        # 3. Spatial OCR extraction
        df_spatial = ocr_image_spatial(preprocessed_image)
        
        # 4. Filter by confidence
        df_filtered = filter_by_confidence(df_spatial, threshold=confidence_threshold)
        
        # 5. Store result
        results[page_num] = df_filtered

        log.info(f"    Page {page_num}: {len(df_filtered)} words with confidence >= {confidence_threshold}%")

        # 6. Clean up
        del image, preprocessed_image
        gc.collect()

    log.info("Spatial OCR complete. %d pages processed.", total_pages)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 70)
        print("PDF OCR Pipeline with Preprocessing & Spatial Extraction")
        print("=" * 70)
        print("\nUsage:")
        print("  python F.py <input.pdf> [--mode <mode>] [--output <output.txt>]")
        print("\nModes:")
        print("  standard    Extract text only (default)")
        print("  spatial     Extract text WITH location & confidence (requires pandas)")
        print("  analyze     Show OCR statistics and confidence distribution")
        print("  hocr        Generate hOCR HTML output (layout-preserving)")
        print("  zones       Analyze document structure and extract by zones")
        print("\nExamples:")
        print("  python F.py document.pdf")
        print("  python F.py document.pdf --mode spatial")
        print("  python F.py document.pdf --mode analyze")
        print("  python F.py document.pdf --mode hocr --output output_dir/")
        print("  python F.py document.pdf --mode zones")
        print("=" * 70)
        sys.exit(1)

    # Parse arguments
    input_pdf = sys.argv[1]
    mode = "standard"
    output_file = Path(input_pdf).stem + "_ocr.txt"
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print(f"\n{'='*70}")
    print(f"Mode: {mode.upper()}")
    print(f"PDF: {input_pdf}")
    print(f"{'='*70}\n")

    if mode == "spatial":
        # Spatial extraction mode
        spatial_data = process_pdf_spatial(input_pdf, confidence_threshold=50)
        
        # Save spatial data to CSV
        csv_output = Path(output_file).stem + "_spatial.csv"
        all_data = []
        for page_num, df in spatial_data.items():
            df['page'] = page_num
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(csv_output, index=False)
        
        # Also create readable text
        text_output = []
        for page_num, df in spatial_data.items():
            text = reconstruct_text_from_spatial(df)
            text_output.append(f"{'='*60}\nPage {page_num}\n{'='*60}\n{text}\n")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(text_output)
        
        print(f"✓ Spatial data saved to: {csv_output}")
        print(f"✓ Text saved to: {output_file}")
        
    elif mode == "analyze":
        # Analysis mode - show statistics
        spatial_data = process_pdf_spatial(input_pdf, confidence_threshold=0)
        
        total_words = sum(len(df) for df in spatial_data.values())
        print(f"\n📊 OCR ANALYSIS RESULTS")
        print(f"{'─'*70}")
        print(f"Total words detected: {total_words}")
        
        for page_num, df in spatial_data.items():
            high_conf = len(df[df['conf'] >= 80])
            med_conf = len(df[(df['conf'] >= 50) & (df['conf'] < 80)])
            low_conf = len(df[df['conf'] < 50])
            avg_conf = df['conf'].mean()
            
            print(f"\nPage {page_num}:")
            print(f"  Words: {len(df)}")
            print(f"  Avg Confidence: {avg_conf:.1f}%")
            print(f"  High (≥80%): {high_conf}  |  Medium (50-80%): {med_conf}  |  Low (<50%): {low_conf}")
        
        # Save analysis to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("OCR CONFIDENCE ANALYSIS\n")
            f.write("="*70 + "\n\n")
            for page_num, df in spatial_data.items():
                f.write(f"Page {page_num}:\n")
                f.write(f"  Total words: {len(df)}\n")
                f.write(f"  Average confidence: {df['conf'].mean():.1f}%\n")
                f.write(f"  Confidence distribution:\n")
                f.write(f"    80-100%: {len(df[df['conf'] >= 80])}\n")
                f.write(f"    50-79%: {len(df[(df['conf'] >= 50) & (df['conf'] < 80)])}\n")
                f.write(f"    <50%: {len(df[df['conf'] < 50])}\n\n")
        
        print(f"\n✓ Analysis saved to: {output_file}")
        
    elif mode == "hocr":
        # hOCR output mode - HTML format with layout preservation
        spatial_data = process_pdf_spatial(input_pdf, confidence_threshold=0)
        
        # Determine output directory
        if output_file.endswith(('.txt', '.html', '.hocr')):
            hocr_dir = Path(output_file).stem + "_hocr"
        else:
            hocr_dir = output_file if output_file != Path(input_pdf).stem + "_ocr.txt" else Path(input_pdf).stem + "_hocr"
        
        # Generate and save hOCR for each page
        save_hocr(spatial_data, hocr_dir)
        
        print(f"\n📄 hOCR Output Generated")
        print(f"{'─'*70}")
        print(f"Output directory: {hocr_dir}/")
        print(f"Files created:")
        for page_num in sorted(spatial_data.keys()):
            print(f"  page_{page_num}.hocr ({len(spatial_data[page_num])} words)")
        print(f"\n💡 hOCR files are HTML documents with layout information preserved.")
        print(f"Can be opened in browsers or processed by document analysis tools.")
        
    elif mode == "zones":
        # Zone-based extraction - identify document structure
        spatial_data = process_pdf_spatial(input_pdf, confidence_threshold=0)
        
        print(f"\n📍 DOCUMENT ZONE ANALYSIS")
        print(f"{'═'*70}")
        
        for page_num, df in spatial_data.items():
            print(f"\n📄 Page {page_num}:")
            print(f"{'─'*70}")
            
            # Create standard zones (assuming 300 DPI page dimensions)
            # Detect actual page dimensions from data
            if len(df) > 0:
                page_width = int(df['left'].max() + df['width'].max())
                page_height = int(df['top'].max() + df['height'].max())
            else:
                page_width = 2000
                page_height = 2600
            
            zones = create_standard_zones(page_width, page_height)
            zone_analysis = analyze_document_structure(df, page_width, page_height)
            
            # Display zone information
            for zone_name in ['header', 'title', 'top_right', 'body', 'footer']:
                stats = zone_analysis.get(zone_name, {})
                word_count = stats.get('word_count', 0)
                avg_conf = stats.get('avg_confidence', 0)
                words_preview = stats.get('words', '')
                
                if word_count > 0:
                    print(f"\n  {zone_name.upper()}: {word_count} words (avg conf: {avg_conf:.0f}%)")
                    print(f"    Preview: {words_preview[:60]}...")
            
            # Show extraction examples
            print(f"\n{'─'*70}")
            print(f"ZONE EXTRACTION EXAMPLES:")
            
            header_zone = zones['header']
            header_text = get_zone_text(df, header_zone, min_confidence=60)
            if header_text.strip():
                print(f"\n  [HEADER - Top 8% of page]")
                print(f"  {header_text[:100].strip()}...")
            
            title_zone = zones['title']
            title_text = get_zone_text(df, title_zone, min_confidence=70)
            if title_text.strip():
                print(f"\n  [TITLE - Below header]")
                print(f"  {title_text[:100].strip()}...")
            
            footer_zone = zones['footer']
            footer_text = get_zone_text(df, footer_zone, min_confidence=50)
            if footer_text.strip():
                print(f"\n  [FOOTER - Bottom 15% of page]")
                print(f"  {footer_text[:100].strip()}...")
        
        # Save zone analysis to file
        if output_file and output_file != Path(input_pdf).stem + "_ocr.txt":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("ZONE-BASED DOCUMENT ANALYSIS\n")
                f.write("="*70 + "\n\n")
                
                for page_num, df in spatial_data.items():
                    zones = create_standard_zones()
                    analysis = analyze_document_structure(df)
                    
                    f.write(f"Page {page_num}:\n")
                    f.write(f"{'─'*70}\n\n")
                    
                    for zone_name, stats in analysis.items():
                        f.write(f"{zone_name.upper()}:\n")
                        f.write(f"  Word count: {stats['word_count']}\n")
                        f.write(f"  Avg confidence: {stats['avg_confidence']:.1f}%\n")
                        f.write(f"  Preview: {stats['words']}\n\n")
            
            print(f"\n✓ Zone analysis saved to: {output_file}")
        
    else:
        # Standard mode (default)
        pages = process_pdf(input_pdf, output_path=output_file)
        print(f"\n✓ Done. Extracted text from {len(pages)} pages → {output_file}")