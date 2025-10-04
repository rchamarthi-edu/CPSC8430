# === IMPORT STATEMENTS ===
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# === CONSTANTS & SPECIAL TOKENS ===
PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"

# === TEXT & DATA PROCESSING ===
def tokenize(text):
    """Cleans and splits text into a list of tokens."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

def load_labels(path):
    """Loads captions from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item["caption"] for item in data}

def load_features(path):
    """Loads video features from .npy files."""
    features = {}
    for feature_file in sorted(path.glob("*.npy")):
        feat = np.load(feature_file)
        if feat.ndim == 1:
            feat = feat[None, :]
        features[feature_file.stem] = feat.astype(np.float32)
    return features

# === METRIC CALCULATION ===
def calculate_bleu(candidate_caption, reference_captions):
    """Calculates sentence-level BLEU-1 score using NLTK."""
    smoothie = SmoothingFunction().method1
    #split captions into lists of words
    candidate = candidate_caption.lower().split()
    references = [ref.lower().split() for ref in reference_captions]
    #calculate bleu-1 score (weights are 1 for unigrams, 0 for others)
    return sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)

# === VOCABULARY CLASS ===
class Vocab:
    """Manages the mapping between tokens and numerical indices."""
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.itos = [PAD, BOS, EOS, UNK]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3  

    def __len__(self):
        return len(self.itos)

    def build(self, all_captions):
        """Builds the vocabulary from a list of captions."""
        token_counts = Counter(token for caption in all_captions for token in tokenize(caption))
        for token, freq in token_counts.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    def encode(self, tokens):
        """Converts a list of tokens to a list of indices."""
        return [self.stoi.get(token, self.stoi[UNK]) for token in tokens]

    def decode(self, ids):
        """Converts a list of indices back to a list of tokens."""
        return [self.itos[i] for i in ids]

# === PYTORCH DATASET SETUP & COLLATE FUNCTION ===
class CaptionDataset(Dataset):
    """PyTorch Dataset for loading video features and captions."""
    def __init__(self, features, labels, vocab, video_ids):
        super().__init__()
        self.features = features
        self.labels = labels
        self.vocab = vocab
        self.video_ids = video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feature = torch.from_numpy(self.features[video_id])
        caption = random.choice(self.labels[video_id])
        token_ids = self.vocab.encode(tokenize(caption))
        y_in, y_out = [self.vocab.bos_id] + token_ids, token_ids + [self.vocab.eos_id]
        return feature, y_in, y_out


def create_collate_fn(pad_id):
    """Creates a collate function for the DataLoader."""
    def collate_fn(batch):
        features, y_ins, y_outs = zip(*batch)
        feat_lengths = torch.tensor([f.size(0) for f in features])
        padded_feats = nn.utils.rnn.pad_sequence(features, batch_first=True)
        padded_y_ins = nn.utils.rnn.pad_sequence([torch.tensor(y) for y in y_ins], batch_first=True, padding_value=pad_id)
        padded_y_outs = nn.utils.rnn.pad_sequence([torch.tensor(y) for y in y_outs], batch_first=True, padding_value=pad_id)
        return {"features": padded_feats, "feat_lengths": feat_lengths, "captions_in": padded_y_ins, "captions_out": padded_y_outs}
    return collate_fn

# === SEQUENCE TO SEQUENCE MODEL ===
class Encoder(nn.Module):
    def __init__(self, feat_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)

    def forward(self, features, feat_lengths):
        packed_features = pack_padded_sequence(features, feat_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_features)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, hidden, cell):
        embedded = self.embedding(tokens)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc(output), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, feat_dim, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(feat_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.config = {"feat_dim": feat_dim, "vocab_size": vocab_size, "embed_size": embed_size, "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}

    def forward(self, feat_seq, feat_len, y_in_ids):
        hidden, cell = self.encoder(feat_seq, feat_len)
        logits, _, _ = self.decoder(y_in_ids, hidden, cell)
        return logits

# === TRAINING & INFERENCE LOGIC ===
def evaluate(model, loader, criterion, vocab_pad_id, device):
    """Runs a single validation epoch for loss and accuracy."""
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            features, feat_lengths, captions_in, captions_out = [b.to(device) for b in batch.values()]
            logits = model(features, feat_lengths, captions_in)
            loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = (captions_out != vocab_pad_id)
            total_correct += ((preds == captions_out) & mask).sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, val_data, optimizer, vocab, device, epochs, batch_size):
    """Main training loop."""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_bleu': []}
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    print("--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, total_correct_train, total_tokens_train = 0, 0, 0
        for batch in train_loader:
            features, feat_lengths, captions_in, captions_out = [b.to(device) for b in batch.values()]
            optimizer.zero_grad()
            logits = model(features, feat_lengths, captions_in)
            loss = criterion(logits.view(-1, logits.size(-1)), captions_out.view(-1))
            loss.backward()

            #clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = (captions_out != vocab.pad_id)
            total_correct_train += ((preds == captions_out) & mask).sum().item()
            total_tokens_train += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (total_correct_train / total_tokens_train) * 100 if total_tokens_train > 0 else 0
        
        #validation for loss and accuracy (fast)
        val_loss, val_acc = evaluate(model, val_loader, criterion, vocab.pad_id, device)
        
        #validation for bleu score (slower, requires generating sentences)
        val_predictions = generate_predictions(model, val_data['features'], vocab, device, batch_size)
        avg_bleu = np.mean([calculate_bleu(pred, val_data['labels'][vid]) for vid, pred in val_predictions.items()])

        history['train_loss'].append(avg_train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['val_bleu'].append(avg_bleu)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val BLEU: {avg_bleu:.4f}")
        
    return history

def greedy_decode(model, features, feat_lengths, vocab, device, max_len=30):
    """Generates captions for a batch of videos."""
    model.eval()
    batch_size = features.size(0)
    with torch.no_grad():
        hidden, cell = model.encoder(features, feat_lengths)
        current_tokens = torch.full((batch_size, 1), vocab.bos_id, dtype=torch.long, device=device)
        generated_sequences = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        
        for t in range(max_len):
            logits, hidden, cell = model.decoder(current_tokens, hidden, cell)
            next_tokens = logits.argmax(dim=-1)  #shape: [batch_size, 1]
            generated_sequences[:, t] = next_tokens.squeeze(-1)  #squeeze last dim to get [batch_size]
            if (next_tokens.squeeze(-1) == vocab.eos_id).all(): break
            current_tokens = next_tokens

    captions = []
    for i in range(batch_size):
        raw_ids = generated_sequences[i].tolist()
        try:
            eos_index = raw_ids.index(vocab.eos_id)
            raw_ids = raw_ids[:eos_index]
        except ValueError: pass
        captions.append(" ".join(vocab.decode(raw_ids)))
        
    return captions

def generate_predictions(model, features, vocab, device, batch_size):
    """Generates captions for all test videos."""
    model.eval()
    predictions = {}
    video_ids = sorted(features.keys())
    for i in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[i:i+batch_size]
        batch_feats_list = [torch.from_numpy(features[vid]).to(device) for vid in batch_ids]
        feat_lengths = torch.tensor([f.size(0) for f in batch_feats_list])
        padded_feats = nn.utils.rnn.pad_sequence(batch_feats_list, batch_first=True)
        captions = greedy_decode(model, padded_feats, feat_lengths, vocab, device)
        for vid, cap in zip(batch_ids, captions): predictions[vid] = cap
    return predictions

def save_training_plots(history, save_path):
    """Saves plots for loss, accuracy, and BLEU score."""
    print(f"Generating and saving training plots to {save_path}")
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Training Performance', fontsize=16, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4); axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4); axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.4)
    
    axes[2].plot(epochs, history['val_bleu'], 'g-o', label='Validation BLEU-1', markersize=4)
    axes[2].set_title('BLEU-1 Score'); axes[2].set_xlabel('Epochs'); axes[2].set_ylabel('BLEU-1'); axes[2].legend(); axes[2].grid(True, alpha=0.4)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(save_path); plt.close()

# === MAIN FUNCTION ===
def main(): 
    parser = argparse.ArgumentParser(description="HW2 - Seq2Seq Model for Video Captioning")
    parser.add_argument("data_dir", type=str, help="Path to data directory")
    parser.add_argument("output_file", type=str, help="Path for output predictions")
    parser.add_argument("--model_path", type=str, default="rc_seq2seq_model")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    
    EPOCHS, BATCH_SIZE, HIDDEN_SIZE, EMBED_SIZE, LEARNING_RATE, DROPOUT, MIN_WORD_FREQ = 200, 64, 256, 256, 0.001, 0.3, 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path(args.model_path)
    is_training_mode = args.train or not model_path.exists()
    
    if is_training_mode:
        print("\n--- Training Mode ---")
        root_dir = Path(args.data_dir)
        train_labels, train_features = load_labels(root_dir / "training_label.json"), load_features(root_dir / "training_data" / "feat")
        
        vocab = Vocab(min_freq=MIN_WORD_FREQ)
        vocab.build([cap for caps in train_labels.values() for cap in caps])
        print(f"Vocabulary built, size: {len(vocab)}")
        
        video_ids = sorted(train_features.keys())
        random.shuffle(video_ids)
        split_idx = int(len(video_ids) * 0.9)
        train_ids, val_ids = video_ids[:split_idx], video_ids[split_idx:]
        
        val_data_for_bleu = {'features': {k: train_features[k] for k in val_ids}, 'labels': {k: train_labels[k] for k in val_ids}}
        
        collate_fn = create_collate_fn(vocab.pad_id)
        train_loader = DataLoader(CaptionDataset(train_features, train_labels, vocab, train_ids), BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(CaptionDataset(train_features, train_labels, vocab, val_ids), BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        feat_dim = next(iter(train_features.values())).shape[1]
        model = Seq2Seq(feat_dim, len(vocab), EMBED_SIZE, HIDDEN_SIZE, 2, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = train_model(model, train_loader, val_loader, val_data_for_bleu, optimizer, vocab, device, EPOCHS, BATCH_SIZE)
        save_training_plots(history, "training_performance.png")
        
        torch.save({'model_state_dict': model.state_dict(), 'vocab_itos': vocab.itos, 'model_config': model.config}, model_path)
        print(f"Model saved to {model_path}")
    
    print("\n--- Inference Mode ---")
    checkpoint = torch.load(model_path, map_location=device)
    
    vocab = Vocab()
    vocab.itos = checkpoint["vocab_itos"]
    vocab.stoi = {token: i for i, token in enumerate(vocab.itos)}
    vocab.pad_id, vocab.bos_id, vocab.eos_id = vocab.stoi[PAD], vocab.stoi[BOS], vocab.stoi[EOS]

    model = Seq2Seq(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    #determine test features path based on mode
    if is_training_mode:
        #training mode: data_dir contains both training_data and testing_data
        test_feat_path = Path(args.data_dir) / "testing_data" / "feat"
    else:
        #inference mode: try multiple possible paths
        test_dir = Path(args.data_dir)
        #check if .npy files are directly in the provided directory
        if any(test_dir.glob("*.npy")):
            test_feat_path = test_dir
        #check for feat subdirectory
        elif (test_dir / "feat").exists():
            test_feat_path = test_dir / "feat"
        #check for testing_data/feat
        elif (test_dir / "testing_data" / "feat").exists():
            test_feat_path = test_dir / "testing_data" / "feat"
        else:
            raise FileNotFoundError(f"Cannot find .npy files in {test_dir}")
    
    test_features = load_features(test_feat_path)
    predictions = generate_predictions(model, test_features, vocab, device, BATCH_SIZE)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for vid_id, cap in sorted(predictions.items()): f.write(f"{vid_id},{cap}\n")
    print(f"Predictions written to {output_path}")

if __name__ == "__main__":
    main()