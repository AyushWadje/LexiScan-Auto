import sys
import os
import json
import re
import random
import nltk

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from lexiscan.ocr.pipeline import process_pdf
except ImportError as e:
    print(f"Error importing process_pdf: {e}")
    sys.exit(1)

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)

def auto_label_sentence(sentence):
    tokens = sentence.split()
    tags = ['O'] * len(tokens)

    # Regex Patterns
    # MONEY: $100, $1.5M, 5000 USD
    money_pattern_dollar_num = re.compile(r'^\$')
    money_pattern_standalone = re.compile(r'^\$[\d,]+(\.\d+)?([KMB])?$')
    money_currency_codes = ['USD', 'EUR', 'GBP']

    # DATE: Month DD YYYY, MM/DD/YYYY, YYYY-MM-DD
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December',
              'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    date_pattern_numeric = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})$')

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # --- MONEY DETECTION ---
        # Case 1: $100 (Split token case handled by tokenizer? Assuming standard split)
        # If tokenizer keeps $100 as one token:
        if money_pattern_standalone.match(token):
            tags[i] = 'B-MONEY'
            i += 1
            continue

        # Case 2: $ 100
        if token == '$' and i + 1 < len(tokens) and re.match(r'^[\d,]+(\.\d+)?$', tokens[i+1]):
            tags[i] = 'B-MONEY'
            tags[i+1] = 'I-MONEY'
            i += 2
            continue

        # Case 3: 100 USD
        if re.match(r'^[\d,]+(\.\d+)?$', token) and i + 1 < len(tokens) and tokens[i+1] in money_currency_codes:
            tags[i] = 'B-MONEY'
            tags[i+1] = 'I-MONEY'
            i += 2
            continue

        # --- DATE DETECTION ---
        # Case 1: Numeric Date
        if date_pattern_numeric.match(token):
            tags[i] = 'B-DATE'
            i += 1
            continue

        # Case 2: Month DD, YYYY
        if token in months:
            # Look ahead for Day
            if i + 1 < len(tokens):
                day_token = tokens[i+1].replace(',', '')
                if re.match(r'^\d{1,2}(st|nd|rd|th)?$', day_token):
                    tags[i] = 'B-DATE'
                    tags[i+1] = 'I-DATE'

                    # Optional Year
                    if i + 2 < len(tokens) and re.match(r'^\d{4}$', tokens[i+2]):
                        tags[i+2] = 'I-DATE'
                        i += 3
                        continue
                    i += 2
                    continue

        i += 1

    return tokens, tags

def generate_training_data(contracts_dir, output_file):
    ensure_nltk_resources()

    if not os.path.exists(contracts_dir):
        print(f"Contracts directory not found: {contracts_dir}")
        return

    pdf_files = [f for f in os.listdir(contracts_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs in {contracts_dir}")

    all_data = []
    total_sentences = 0
    total_dates = 0
    total_money = 0
    processed_pdfs = 0
    failed_pdfs = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(contracts_dir, pdf_file)
        print(f"Processing {pdf_file}...")

        try:
            ocr_results = process_pdf(pdf_path)
            processed_pdfs += 1

            for page_text in ocr_results.values():
                sentences = nltk.sent_tokenize(page_text)

                for sentence in sentences:
                    # Clean sentence
                    sentence = sentence.replace('\n', ' ').strip()
                    if not sentence:
                        continue

                    tokens, tags = auto_label_sentence(sentence)

                    # Check if sentence has entities
                    has_entity = any(t != 'O' for t in tags)

                    if has_entity:
                        all_data.append({'tokens': tokens, 'tags': tags})
                        total_dates += tags.count('B-DATE')
                        total_money += tags.count('B-MONEY')
                        total_sentences += 1
                    else:
                        # Keep 30% of negatives
                        if random.random() < 0.3:
                            all_data.append({'tokens': tokens, 'tags': tags})
                            total_sentences += 1

        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
            failed_pdfs.append((pdf_file, str(e)))

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Total Sentences Generated: {total_sentences}")
    print(f"Total DATE Entities: {total_dates}")
    print(f"Total MONEY Entities: {total_money}")
    print(f"Unique PDFs Processed: {processed_pdfs}")
    if failed_pdfs:
        print(f"\nFailed PDFs ({len(failed_pdfs)}):")
        for f, err in failed_pdfs:
            print(f"  - {f}: {err}")
    print(f"\nData saved to: {output_file}")

if __name__ == "__main__":
    CONTRACTS_DIR = os.path.join(root_dir, 'data', 'contracts')
    OUTPUT_FILE = os.path.join(root_dir, 'data', 'training_data.json')

    # Create data/contracts if not exists for testing
    if not os.path.exists(CONTRACTS_DIR):
        os.makedirs(CONTRACTS_DIR)
        print(f"Created {CONTRACTS_DIR}. Please add PDFs there.")

    generate_training_data(CONTRACTS_DIR, OUTPUT_FILE)
