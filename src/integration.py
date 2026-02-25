import sys
import os

# Add project root and PDF directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pdf_dir = os.path.join(root_dir, 'PDF')

if root_dir not in sys.path:
    sys.path.append(root_dir)
if pdf_dir not in sys.path:
    sys.path.append(pdf_dir)

try:
    from F import process_pdf
except ImportError as e:
    print(f"Error importing process_pdf from {pdf_dir}: {e}")
    print("Ensure F.py exists in the PDF directory.")
    sys.exit(1)

from src.ner_model import train_ner_model, predict_ner, normalize_text

def extract_entities_from_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")

    # Step 1: OCR
    print("\n--- Step 1: Running OCR ---")
    try:
        # process_pdf returns {page_num: text}
        ocr_results = process_pdf(pdf_path)
    except Exception as e:
        print(f"Error during OCR: {e}")
        return {}

    if not ocr_results:
        print("No text extracted from PDF.")
        return {}

    print(f"OCR extracted text from {len(ocr_results)} pages.")

    # Step 2: NER Model (Train on sample data for now since we don't have weights)
    print("\n--- Step 2: Training NER model (Mock Training) ---")
    try:
        # Use fewer epochs for demo purposes
        model, token2idx = train_ner_model(epochs=1)
    except Exception as e:
        print(f"Error training NER model: {e}")
        return {}

    extracted_entities = {}

    print("\n--- Step 3: Extracting Entities ---")
    for page_num, text in ocr_results.items():
        print(f"Analyzing page {page_num}...")

        # Split by periods to approximate sentences, as NER works best on sentence level
        sentences = text.split('.')
        page_entities = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            normalized_sent = normalize_text(sentence)
            if not normalized_sent:
                continue

            # Predict
            # Note: Max length in model is 20, so long sentences will be truncated
            predictions = predict_ner(normalized_sent, model, token2idx)

            # Extract entities from BIO tags
            current_entity = []
            current_tag = None

            for token, tag in predictions:
                if tag.startswith("B-"):
                    if current_entity:
                        page_entities.append((" ".join(current_entity), current_tag))
                    current_entity = [token]
                    current_tag = tag[2:]
                elif tag.startswith("I-") and current_tag == tag[2:]:
                    current_entity.append(token)
                else:
                    if current_entity:
                        page_entities.append((" ".join(current_entity), current_tag))
                    current_entity = []
                    current_tag = None

            if current_entity:
                 page_entities.append((" ".join(current_entity), current_tag))

        extracted_entities[page_num] = page_entities

    return extracted_entities

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)

        results = extract_entities_from_pdf(pdf_path)

        print("\n" + "="*50)
        print("FINAL EXTRACTION RESULTS")
        print("="*50)

        for page, entities in results.items():
            print(f"\nPage {page}:")
            if not entities:
                print("  (No entities found)")

            # Dedup within page
            unique_entities = sorted(list(set(entities)))
            for text, label in unique_entities:
                print(f"  [{label}] {text}")
    else:
        print("Usage: python src/integration.py <path_to_pdf>")
