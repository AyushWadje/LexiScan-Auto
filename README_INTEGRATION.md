# Integrated OCR and NER Pipeline

This directory contains the integration of the OCR pipeline (`PDF/F.py`) and the Named Entity Recognition (NER) model (`src/ner_model.py`).

## Structure

- `src/ner_model.py`: Contains the Bi-LSTM NER model definition, data preparation, and training/prediction logic.
- `src/integration.py`: Main script that combines OCR and NER to extract entities from PDF files.
- `PDF/F.py`: The OCR pipeline using Tesseract and PDF2Image.

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
    -   Ensure both are in your system PATH or configured in `PDF/F.py`.

3.  **Python Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To extract entities from a PDF file:

```bash
python src/integration.py path/to/document.pdf
```

## How it works

1.  **OCR**: The script uses `pdf2image` to convert PDF pages to images, then preprocesses them (grayscale, binarization, deskewing) using OpenCV, and extracts text using Tesseract.
2.  **NER**: The text is normalized and passed to the Bi-LSTM NER model.
    -   *Note*: Currently, the model is trained on a small sample dataset included in `src/ner_model.py` each time you run the script. In a production environment, you would load pre-trained weights.
3.  **Extraction**: The model identifies `DATE` and `MONEY` entities, which are then printed to the console.

## Configuration

-   **OCR Settings**: Modified in `PDF/F.py`.
-   **Model Settings**: Modified in `src/ner_model.py` (e.g., hyperparameters, training data).
