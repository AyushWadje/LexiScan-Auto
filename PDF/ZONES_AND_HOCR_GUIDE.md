# Zone-Based Extraction & hOCR
## Advanced Document Analysis and Layout Preservation

---

## Part 1: Zone-Based Extraction

### What Are Zones?

Zones are **rectangular regions on a page** defined by coordinates. Instead of processing the entire document, zones allow you to extract specific areas intelligently.

```
┌─────────────────────────────────────────────┐
│         HEADER (Top 8%)                     │
├─────────────────────────────────────────────┤
│         TITLE (8-20%)                       │
├──────────────────────────┬──────────────────┤
│  BODY (20-85%)           │ TOP_RIGHT (0-15%)│
│                          │                  │
│                          │                  │
├──────────────────────────┴──────────────────┤
│         FOOTER (Bottom 15%)                 │
└─────────────────────────────────────────────┘
```

### Standard Zones

Your system defines these zones automatically (based on 300 DPI A4/Letter page):

| Zone | Purpose | Y Position | Content |
|------|---------|-----------|---------|
| **Header** | Document metadata, dates | Top 0-8% | Version, date, document ID |
| **Title** | Document title/subject | 8-20% | "Employment Contract", "Invoice #001" |
| **Top Right** | Special fields | Top 0-15%, Right 60%+ | Signature, date, reference |
| **Body** | Main content | 20-85% | Terms, conditions, details |
| **Footer** | Page footer | Bottom 85-100% | Page numbers, copyright, contact |
| **Left Margin** | Left side | Any Y, X < 10% | Annotations, page markers |
| **Right Margin** | Right side | Any Y, X > 90% | Binding margin, notes |

### Why Zones Matter for Legal Contracts

#### Example: Extracting Effective Date

A legal contract typically has this structure:
```
══════════════════════════════════════
   CONTRACT AGREEMENT
══════════════════════════════════════

This Agreement is entered into as of the 15th day of February, 2026
(Effective Date typically in HEADER or TITLE zone)

TERMS AND CONDITIONS:
  Section 1: Definitions...
  Section 2: Services...
  (BODY zone - main content)

SIGNATURES:
_________________      __________________
Company Signature      Client Signature
Date: __________      Date: __________
(FOOTER zone - signature block)
```

**Smart extraction logic:**
```python
# Get all dates from header
header_dates = extract_dates_from_zone(df, header_zone)
# Very high probability this is the effective date

# Get all dates from footer  
footer_dates = extract_dates_from_zone(df, footer_zone)
# These are likely signature dates

# Dates in body? Check context
body_dates = extract_dates_from_zone(df, body_zone)
# Likely references to dates in terms
```

### Using Zone-Based Extraction

#### Basic Usage
```bash
python F.py "contract.pdf" --mode zones
```

Output shows:
```
HEADER: 10 words (avg conf: 85%)
  Preview: CONTRACT AGREEMENT EXECUTED...

TITLE: 27 words (avg conf: 82%)
  Preview: This Agreement is entered into...

BODY: 295 words (avg conf: 75%)
  Preview: TERMS AND CONDITIONS Section 1...

FOOTER: 45 words (avg conf: 88%)
  Preview: Signature lines Date signature...
```

#### Python API - Extracting Specific Zones
```python
from F import process_pdf_spatial, create_standard_zones, extract_zone, get_zone_text

# Get spatial data
spatial_data = process_pdf_spatial("contract.pdf")

for page_num, df in spatial_data.items():
    # Create zones
    zones = create_standard_zones()
    
    # Extract header text (high confidence only)
    header_text = get_zone_text(df, zones['header'], min_confidence=80)
    print(f"Header:\n{header_text}\n")
    
    # Extract signature zone
    footer_text = get_zone_text(df, zones['footer'], min_confidence=70)
    print(f"Signature Block:\n{footer_text}\n")
    
    # Extract body
    body_text = get_zone_text(df, zones['body'], min_confidence=60)
    print(f"Contract Terms:\n{body_text[:500]}...")
```

#### Custom Zones
```python
from F import DocumentZone, extract_zone

# Define a custom zone for "Signature Area" (bottom-right)
signature_zone = DocumentZone(
    name='Signature',
    x_min=1200,  # Right side
    x_max=2000,
    y_min=2300,  # Bottom 300 pixels
    y_max=2600
)

# Extract all words in that area
sig_words = extract_zone(df, signature_zone)
print(f"Signature words: {sig_words['text'].tolist()}")
```

#### Multi-Zone Analysis
```python
from F import process_pdf_spatial, create_standard_zones, extract_all_zones

spatial_data = process_pdf_spatial("document.pdf")

for page_num, df in spatial_data.items():
    zones = create_standard_zones()
    
    # Get all zones at once
    all_zones = extract_all_zones(df, zones)
    
    # Analyze confidence per zone
    for zone_name, zone_df in all_zones.items():
        if len(zone_df) > 0:
            avg_conf = zone_df['conf'].mean()
            high_conf = len(zone_df[zone_df['conf'] >= 80])
            print(f"{zone_name}: {avg_conf:.0f}% avg, {high_conf} high-confidence words")
```

### Real-World Example: Invoice Extraction

```python
from F import process_pdf_spatial, create_standard_zones, get_zone_text

spatial_data = process_pdf_spatial("invoice.pdf")

for page_num, df in spatial_data.items():
    zones = create_standard_zones()
    
    # Extract invoice number from header
    header = get_zone_text(df, zones['header'], min_confidence=80)
    # RegEx: match "Invoice #\d+"
    
    # Extract company info from top-left
    top_left = DocumentZone('TopLeft', 0, 800, 0, 400)
    company_text = get_zone_text(df, top_left, min_confidence=75)
    
    # Extract line items from body
    body = get_zone_text(df, zones['body'], min_confidence=70)
    
    # Extract total from bottom-right
    bottom_right = DocumentZone('BottomRight', 1200, 2000, 2200, 2600)
    total_text = get_zone_text(df, bottom_right, min_confidence=80)
```

---

## Part 2: hOCR (HTML-based OCR)

### What is hOCR?

hOCR is an **HTML/XML format for OCR output** that preserves:
- **Layout information** (bounding boxes)
- **Confidence scores** (per word)
- **Document hierarchy** (pages → blocks → paragraphs → lines → words)

It's the **industry standard** used by:
- Legal document processors
- E-discovery software
- PDF highlighters
- Document management systems

### hOCR Structure

Each recognized word is marked with:
- **bbox**: Bounding box coordinates (x1 y1 x2 y2)
- **x_size**: Estimated font size (in pixels)
- **x_conf**: Confidence score (0-100)

#### Example hOCR:
```html
<span class="ocrx_word" 
      id="word_1_1_1_1_1" 
      title="bbox 850 154 1068 197; x_size 43; x_conf 85">
  AYUSH
</span>
```

Breakdown:
```
bbox 850 154 1068 197
  ↓
  Left=850px, Top=154px, Right=1068px, Bottom=197px
  (Word is 218 pixels wide, 43 pixels tall)

x_size 43
  ↓ Font height estimate: 43 pixels (≈ 10.75pt @ 300 DPI)

x_conf 85
  ↓ Tesseract is 85% confident this word is correct
```

### Generating hOCR

#### Basic Usage
```bash
python F.py "document.pdf" --mode hocr --output "output_dir"
```

Creates: `output_dir/page_1.hocr`, `output_dir/page_2.hocr`, etc.

#### In Python
```python
from F import process_pdf_spatial, save_hocr

# Extract spatial data
spatial_data = process_pdf_spatial("document.pdf")

# Generate hOCR files
save_hocr(spatial_data, output_path="hocr_files")
# Creates: hocr_files/page_1.hocr, page_2.hocr, ...
```

### Reading and Using hOCR Files

#### In a Browser
```bash
# Simply open in Chrome/Firefox
open hocr_output/page_1.hocr

# Or drag into browser window
```

The HTML is readable and shows the document text with metadata.

#### Parsing with Python
```python
from lxml import etree

# Load hOCR file
hocr = etree.parse("page_1.hocr")
root = hocr.getroot()

# Find all words
words = root.findall(".//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")

for word_elem in words:
    word_id = word_elem.get('id')
    word_text = word_elem.text
    title = word_elem.get('title')
    
    # Parse bbox from title attribute
    # "bbox 850 154 1068 197; x_size 43; x_conf 85"
    parts = title.split(';')
    bbox_str = parts[0].replace('bbox ', '')
    left, top, right, bottom = map(int, bbox_str.split())
    
    print(f"{word_text}: ({left}, {top}) - ({right}, {bottom})")
```

#### JavaScript - Highlight Words in Browser
```javascript
// Open hOCR in browser, then run:
var words = document.querySelectorAll('[class="ocrx_word"]');

// Highlight high-confidence words (green) and low-confidence (red)
words.forEach(function(word) {
    var title = word.getAttribute('title');
    var conf = title.match(/x_conf (\d+)/)[1];
    
    if (conf >= 80) {
        word.style.backgroundColor = '#90EE90';  // Light green
    } else if (conf >= 50) {
        word.style.backgroundColor = '#FFD700';  // Gold
    } else {
        word.style.backgroundColor = '#FFB6C6';  // Light red
    }
});
```

### hOCR in Professional Workflows

#### 1. Legal Discovery (e-discovery)
```
Original PDF ──→ OCR ──→ hOCR ──→ Search/Highlight
                         ├─ Preserves layout
                         ├─ Searchable text
                         └─ Clickable = jump to PDF location
```

#### 2. Accessibility
```python
# Generate hOCR, then create accessible PDF with text layer
hocr_data = extract_hocr("document.hocr")
accessible_pdf = add_text_layer_to_pdf("original.pdf", hocr_data)
# Now PDF is searchable and has alt-text
```

#### 3. Validation/Quality Control
```python
from lxml import etree

def ocr_quality_report(hocr_file):
    hocr = etree.parse(hocr_file)
    
    # Count words by confidence
    words = hocr.findall(".//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
    
    high = sum(1 for w in words if int(w.get('title').split('x_conf ')[1].split(';')[0]) >= 80)
    med = sum(1 for w in words if 50 <= int(...) < 80)
    low = sum(1 for w in words if int(...) < 50)
    
    print(f"Quality Report: {high} high, {med} med, {low} low")
    
    if high / len(words) < 0.5:
        print("⚠ WARNING: Low OCR quality, may need manual review")
```

### hOCR vs. Other Formats

| Format | Pros | Cons | Use Case |
|--------|------|------|----------|
| **hOCR** | Standard, layout-preserving, XML-parseable | Slightly verbose | Legal, professional |
| **CSV (Spatial)** | Simple, easy to filter, pandas-friendly | No hierarchy, no stylesheet | Data analysis |
| **Plain Text** | Simplest, human-readable | No location data | Quick reading |
| **PDF with Text Layer** | Visually preserves original | Large file size | Accessibility |

---

## Combining Zones + hOCR

### Use Case: Smart Contract Analysis

```python
from F import process_pdf_spatial, create_standard_zones, save_hocr, get_zone_text

# Process contract
spatial = process_pdf_spatial("contract.pdf", confidence_threshold=50)

# Generate hOCR (for highlighting)
save_hocr(spatial, "contract_hocr")

# Extract key zones
for page_num, df in spatial.items():
    zones = create_standard_zones()
    
    # Get structured data
    signature_zone = DocumentZone('Signature', 1200, 2000, 2400, 2600)
    sig_text = get_zone_text(df, signature_zone, min_confidence=80)
    
    # hOCR has all the details
    # Use it to highlight signature area in UI
    print(f"High-confidence signature words in hOCR: contract_hocr/page_{page_num}.hocr")
    print(f"Jump to bbox coordinates to locate in PDF")
```

---

## Summary

- **Zones**: Extract specific regions (header, footer, signature area)
- **hOCR**: Layout-preserving HTML format with coordinates and confidence
- **Together**: Intelligent field extraction + professional document processing

Your system now supports:
✅ Zone-based intelligent extraction
✅ Professional hOCR HTML generation  
✅ Document structure analysis
✅ Integration with external tools

