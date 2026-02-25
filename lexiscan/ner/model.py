
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout, Input, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import f1_score
import os
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# BIO Tags
BIO_TAGS = ['O', 'B-DATE', 'I-DATE', 'B-MONEY', 'I-MONEY']
tag2idx = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Sample Data (Positive and Negative)
SAMPLE_DATA = [
    # POSITIVE EXAMPLES (DATE/MONEY)
    (['The', 'meeting', 'is', 'on', 'January', '15', '2024'], ['O', 'O', 'O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Price', 'is', '$', '50'], ['O', 'O', 'B-MONEY', 'I-MONEY']),
    (['Due', 'date', 'is', 'March', '1'], ['O', 'O', 'O', 'B-DATE', 'I-DATE']),
    (['Cost', '$', '100', 'on', 'Feb', '28', '2024'], ['O', 'B-MONEY', 'I-MONEY', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Deadline', 'April', '10'], ['O', 'B-DATE', 'I-DATE']),
    (['The', 'contract', 'value', 'is', '$', '5000'], ['O', 'O', 'O', 'O', 'B-MONEY', 'I-MONEY']),
    (['Submit', 'by', 'December', '31', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Payment', 'of', '$', '1500', 'due', 'June', '1'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'B-DATE', 'I-DATE']),
    (['Invoice', 'dated', 'March', '15', '2023'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Budget', 'approved', '$', '250', 'on', 'July', '4'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'B-DATE', 'I-DATE']),
    (['Agreement', 'effective', 'September', '1', '2023'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Payment', 'amount', '$', '7500', 'USD'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'I-MONEY']),
    (['Contract', 'expires', 'on', 'October', '31', '2024'], ['O', 'O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Total', 'cost', '$', '15000'], ['O', 'O', 'B-MONEY', 'I-MONEY']),
    (['Meeting', 'scheduled', 'for', 'November', '5'], ['O', 'O', 'O', 'B-DATE', 'I-DATE']),
    (['Invoice', 'amount', '$', '3200', 'due', 'immediately'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'O']),
    (['The', 'project', 'starts', 'on', 'January', '10', '2025'], ['O', 'O', 'O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Fee', 'of', '$', '500', 'per', 'month'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'O']),
    (['Warranty', 'valid', 'until', 'December', '15', '2023'], ['O', 'O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Deposit', '$', '1000', 'required'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Effective', 'date', 'is', 'February', '1'], ['O', 'O', 'O', 'B-DATE', 'I-DATE']),
    (['Budget', 'allocation', '$', '50000', 'annually'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Deadline', 'is', 'May', '30', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Amount', 'paid', '$', '2500', 'on', 'date'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'O']),
    (['Event', 'on', 'August', '20'], ['O', 'O', 'B-DATE', 'I-DATE']),
    (['Price', 'tag', '$', '899', 'per', 'unit'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O', 'O']),
    (['Anniversary', 'date', 'June', '15', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Contract', 'value', '$', '25000', 'total'], ['O', 'O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Action', 'date', 'March', '10', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Premium', '$', '450', 'monthly'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Conference', 'on', 'July', '21', '2023'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Investment', '$', '100000', 'required'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Launch', 'date', 'April', '1'], ['O', 'O', 'B-DATE', 'I-DATE']),
    (['Cost', 'estimate', '$', '12000'], ['O', 'O', 'B-MONEY', 'I-MONEY']),
    (['Opening', 'on', 'September', '15', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Salary', '$', '80000', 'annually'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Check', 'in', 'date', 'November', '1'], ['O', 'O', 'O', 'B-DATE', 'I-DATE']),
    (['Rent', '$', '1500', 'monthly'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Review', 'scheduled', 'January', '25', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Fine', '$', '200', 'payment'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Closing', 'date', 'May', '15', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Bonus', '$', '5000', 'awarded'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Termination', 'date', 'December', '31', '2023'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Settlement', '$', '75000', 'total'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Renewal', 'on', 'August', '10'], ['O', 'O', 'B-DATE', 'I-DATE']),
    (['Invoice', '$', '4500', 'payable'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Publication', 'date', 'June', '30', '2024'], ['O', 'O', 'B-DATE', 'I-DATE', 'I-DATE']),
    (['Rebate', '$', '300', 'available'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    (['Trial', 'period', 'starts', 'February', '14'], ['O', 'O', 'O', 'B-DATE', 'I-DATE']),
    (['Commission', '$', '10000', 'earned'], ['O', 'B-MONEY', 'I-MONEY', 'O']),
    # NEGATIVE EXAMPLES (no DATE/MONEY)
    (['The', 'document', 'is', 'signed'], ['O', 'O', 'O', 'O']),
    (['Please', 'review', 'the', 'agreement'], ['O', 'O', 'O', 'O']),
    (['We', 'confirm', 'the', 'terms'], ['O', 'O', 'O', 'O']),
    (['This', 'is', 'a', 'contract'], ['O', 'O', 'O', 'O']),
    (['All', 'parties', 'agree'], ['O', 'O', 'O']),
    (['The', 'agreement', 'is', 'final'], ['O', 'O', 'O', 'O']),
    (['We', 'need', 'your', 'approval'], ['O', 'O', 'O', 'O']),
    (['Please', 'sign', 'here'], ['O', 'O', 'O']),
    (['The', 'terms', 'apply'], ['O', 'O', 'O']),
    (['This', 'is', 'binding'], ['O', 'O', 'O']),
]

MAX_SEQ_LENGTH = 20

# GloVe Download and Load
def load_glove_embeddings(embedding_dim=100):
    try:
        import kagglehub
        print("Downloading GloVe embeddings from Kaggle...")
        glove_path = kagglehub.dataset_download("danielwillgeorge/glove6b100dtxt")
        glove_file = os.path.join(glove_path, "glove.6B.100d.txt")

        embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embedding_dim:
                    embeddings[word] = vector
        print(f"✅ Loaded {len(embeddings):,} GloVe vectors (100D)")
        return embeddings
    except Exception as e:
        print(f"⚠️ Could not load GloVe embeddings: {e}")
        print("   Using random embeddings instead.")
        return {}

def build_vocabulary(data):
    all_tokens = set()
    for sent_tokens, _ in data:
        all_tokens.update(sent_tokens)

    token2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, token in enumerate(sorted(all_tokens), start=2):
        token2idx[token] = idx

    return token2idx

def normalize_text(text):
    """Normalize text: lowercase, remove extra spaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """Simple whitespace tokenization."""
    return text.split()

def prepare_sequences(data, token2idx, tag2idx, max_length=20):
    X, y = [], []
    for tokens, tags in data:
        token_ids = [token2idx.get(token, token2idx['<UNK>']) for token in tokens]
        tag_ids = [tag2idx[tag] for tag in tags]
        X.append(token_ids)
        y.append(tag_ids)
    X_padded = pad_sequences(X, maxlen=max_length, padding='post', value=token2idx['<PAD>'])
    y_padded = pad_sequences(y, maxlen=max_length, padding='post', value=tag2idx['O'])
    return X_padded, y_padded

def build_embedding_matrix(token2idx, glove_embeddings, embedding_dim=100):
    vocab_size = len(token2idx)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)
    embedding_matrix[token2idx['<PAD>']] = np.zeros(embedding_dim, dtype=np.float32)

    if not glove_embeddings:
        return embedding_matrix

    all_vectors = np.array(list(glove_embeddings.values()), dtype=np.float32)
    unk_vector = all_vectors.mean(axis=0).astype(np.float32)
    embedding_matrix[token2idx['<UNK>']] = unk_vector

    found = 0
    for token, idx in token2idx.items():
        if token in ('<PAD>', '<UNK>'):
            continue
        token_lower = token.lower()
        if token_lower in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[token_lower]
            found += 1
        else:
            embedding_matrix[idx] = unk_vector

    print(f"Embedding matrix: {embedding_matrix.shape}")
    print(f"  GloVe hits: {found}")
    return embedding_matrix

class F1EarlyStopping(Callback):
    def __init__(self, validation_data, patience=10):
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.best_f1 = 0.0
        self.wait = 0
        self.best_weights = None
        self.history_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=-1)
        y_true_flat, y_pred_flat = [], []
        for i in range(len(y_val)):
            for j in range(len(y_val[i])):
                true_tag = y_val[i][j]
                pred_tag = y_pred[i][j]
                if true_tag != tag2idx['O'] or pred_tag != tag2idx['O']:
                    y_true_flat.append(1 if true_tag != tag2idx['O'] else 0)
                    y_pred_flat.append(1 if pred_tag != tag2idx['O'] else 0)
        current_f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0) if y_true_flat else 0.0
        self.history_f1.append(current_f1)

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n    Early stop at epoch {epoch+1}. Best F1: {self.best_f1:.4f}")
                self.model.stop_training = True
                if self.best_weights:
                    self.model.set_weights(self.best_weights)

def build_improved_bilstm(vocab_size, embedding_dim, embedding_matrix, lstm_units, num_tags, max_length):
    inputs = Input(shape=(max_length,), name='token_input')
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        mask_zero=True,
        name='glove_embedding'
    )(inputs)
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_1'
    )(x)
    x = Dropout(0.3, name='dropout_1')(x)
    x = Bidirectional(
        LSTM(lstm_units // 2, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_2'
    )(x)
    x = Dropout(0.3, name='dropout_2')(x)
    x = TimeDistributed(Dense(64, activation='relu'), name='dense_hidden')(x)
    x = Dropout(0.2, name='dropout_3')(x)
    outputs = TimeDistributed(Dense(num_tags, activation='softmax'), name='ner_output')(x)
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_NER')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_sample_weights(y_padded, class_weights):
    weights = np.ones_like(y_padded, dtype=np.float32)
    for tag_idx, weight in class_weights.items():
        weights[y_padded == tag_idx] = weight
    return weights

def train_ner_model(sample_data=SAMPLE_DATA, epochs=5):
    print("Preparing data...")
    token2idx = build_vocabulary(sample_data)
    vocab_size = len(token2idx)

    # Try loading GloVe, or use random init
    glove_embeddings = load_glove_embeddings()
    embedding_matrix = build_embedding_matrix(token2idx, glove_embeddings)

    X_all, y_all = prepare_sequences(sample_data, token2idx, tag2idx, MAX_SEQ_LENGTH)

    train_split = int(0.7 * len(X_all))
    val_split = int(0.85 * len(X_all))

    X_train, y_train = X_all[:train_split], y_all[:train_split]
    X_val, y_val = X_all[train_split:val_split], y_all[train_split:val_split]

    CLASS_WEIGHTS = {
        tag2idx['O']:       0.3,
        tag2idx['B-DATE']:  3.0,
        tag2idx['I-DATE']:  2.5,
        tag2idx['B-MONEY']: 3.0,
        tag2idx['I-MONEY']: 2.5,
    }

    sw_train = build_sample_weights(y_train, CLASS_WEIGHTS)

    print("Building model...")
    model = build_improved_bilstm(vocab_size, 100, embedding_matrix, 64, len(BIO_TAGS), MAX_SEQ_LENGTH)

    print(f"Training Phase 1 (Warmup) for {epochs} epochs...")
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        epochs=epochs,
        batch_size=2,
        validation_data=(X_val, y_val),
        verbose=1
    )

    print(f"Training Phase 2 (Fine-tuning) for {epochs} epochs...")
    model.get_layer('glove_embedding').trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        epochs=epochs,
        batch_size=2,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model, token2idx

def save_ner_model(model, token2idx, filepath='ner_model_weights.h5', vocab_path='vocab.json'):
    import json
    model.save_weights(filepath)
    with open(vocab_path, 'w') as f:
        json.dump(token2idx, f)
    print(f"Model saved to {filepath}, vocab to {vocab_path}")

def load_ner_model(filepath='ner_model_weights.h5', vocab_path='vocab.json'):
    import json
    with open(vocab_path, 'r') as f:
        token2idx = json.load(f)
    vocab_size = len(token2idx)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, 100)).astype(np.float32)
    model = build_improved_bilstm(vocab_size, 100, embedding_matrix, 64, len(BIO_TAGS), MAX_SEQ_LENGTH)
    model.load_weights(filepath)
    print(f"Model loaded from {filepath}")
    return model, token2idx

def predict_ner(text, model, token2idx, max_length=20, threshold=0.5):
    tokens = text.split()
    token_ids = [token2idx.get(token, token2idx['<UNK>']) for token in tokens]

    # Pad to max_length for prediction
    padded_ids = pad_sequences([token_ids], maxlen=max_length, padding='post', value=token2idx['<PAD>'])

    probs = model.predict(padded_ids, verbose=0)[0]

    result = []
    for i, token in enumerate(tokens):
        if i >= max_length:
            break # Truncate if longer than max_length for now

        max_prob = np.max(probs[i])
        if max_prob >= threshold:
            tag_idx = np.argmax(probs[i])
        else:
            tag_idx = tag2idx['O']

        result.append((token, idx2tag[tag_idx]))

    return result

if __name__ == "__main__":
    model, token2idx = train_ner_model()

    test_text = "Payment of $ 5000 due on January 15 2024"
    print(f"\nTest prediction: {test_text}")
    predictions = predict_ner(test_text, model, token2idx)
    for token, tag in predictions:
        print(f"{token}: {tag}")
