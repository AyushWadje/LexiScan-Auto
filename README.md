# LexiScan-Auto

LexiScan-Auto is an automated system for extracting entities (Dates and Money) from legal documents (PDFs) using OCR and Named Entity Recognition (NER).

## Project Structure

The repository is organized as follows:

-   `lexiscan/`: Main python package.
    -   `ocr/`: Optical Character Recognition module.
        -   `pipeline.py`: Main OCR pipeline using Tesseract and PDF2Image.
    -   `ner/`: Named Entity Recognition module.
        -   `model.py`: Bi-LSTM NER model definition and training logic.
-   `docs/`: Documentation files.
    -   `ocr/`: OCR specific documentation.
    -   `WEEK2_DOCUMENTATION.md`, etc.
-   `notebooks/`: Jupyter notebooks for experiments (e.g., `Week2_NER_Model.ipynb`).
-   `scripts/`: Utility scripts and demos (e.g., `spatial_ocr_demo.py`).
-   `data/`: Data storage (e.g., hOCR output).
-   `main.py`: Main entry point to run the integrated pipeline.
-   `requirements.txt`: Project dependencies.

## Prerequisites

1.  **Python 3.10+**
2.  **System Dependencies**:
    -   `tesseract-ocr` (and language packs if needed)
    -   `poppler-utils` (for PDF processing)

    On Ubuntu/Debian:
    ```bash
    sudo apt-get install tesseract-ocr poppler-utils
    ```

    On Windows:
    -   Download and install Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
    -   Download and install Poppler from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
    -   Ensure both are in your system PATH.

3.  **Python Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To extract entities from a PDF file:

```bash
python main.py path/to/document.pdf
```

## How it works

1.  **OCR**: The script uses `pdf2image` to convert PDF pages to images, then preprocesses them (grayscale, binarization, deskewing) using OpenCV, and extracts text using Tesseract.
2.  **NER**: The text is normalized and passed to the Bi-LSTM NER model.
    -   *Note*: Currently, the model is trained on a small sample dataset included in `lexiscan/ner/model.py` each time you run the script. In a production environment, you would load pre-trained weights.
3.  **Extraction**: The model identifies `DATE` and `MONEY` entities, which are then printed to the console.
