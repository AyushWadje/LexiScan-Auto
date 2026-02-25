import pytesseract
import os

# Set Tesseract environment variables
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("Tesseract path set to:", pytesseract.pytesseract.pytesseract_cmd)
print("Path exists:", os.path.exists(pytesseract.pytesseract.pytesseract_cmd))
print("TESSDATA_PREFIX:", os.environ.get('TESSDATA_PREFIX'))
print("Attempting to get Tesseract languages...")

try:
    languages = pytesseract.get_languages()
    print("Success! Languages found:", languages)
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")
