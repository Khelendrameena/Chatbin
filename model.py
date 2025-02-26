
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
import sentencepiece as spm  # New: Using SentencePiece for tokenization

# Set Random Seed for Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load SentencePiece Model (or Train it)
SPM_MODEL = "chatbot.model"
try:
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
except:
    spm.SentencePieceTrainer.train(input="vocab.txt", model_prefix="chatbot", vocab_size=8000)
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

vocab_size = sp.vocab_size()
pad_idx = sp.pad_id()
sos_idx = sp.bos_id()
eos_idx = sp.eos_id()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class TransformerChatbot(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_layers=6, ffn_hidden=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, ffn_hidden, dropout, batch_first=True),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_head, ffn_hidden, dropout, batch_first=True),
            num_layers
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg):
        src, trg = self.embedding(src), self.embedding(trg)
        src, trg = self.pos_enc(src), self.pos_enc(trg)
        memory = self.encoder(src)
        output = self.decoder(trg, memory)
        return self.out(output)

# Model & Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChatbot().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load Training Data
with open("data.json", "r") as f:
    data = json.load(f)

# Training Function
def train_model(epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for qus, ans in data:
            src = torch.tensor(sp.encode(qus, out_type=int), dtype=torch.long).unsqueeze(0).to(device)
            trg = torch.tensor([sos_idx] + sp.encode(ans, out_type=int) + [eos_idx], dtype=torch.long).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])  # Exclude last token from target

            loss = criterion(output.reshape(-1, vocab_size), trg[:, 1:].reshape(-1))  # Shift target right
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(data):.4f}")

    torch.save(model.state_dict(), "chatbot_model.pth")

# Beam Search for Response Generation
def beam_search(src, max_len=50, beam_width=5):
    model.eval()
    src = torch.tensor(sp.encode(src, out_type=int), dtype=torch.long).unsqueeze(0).to(device)
    memory = model.encoder(model.embedding(src))
    
    sequences = [[sos_idx]]
    scores = [0]

    for _ in range(max_len):
        all_candidates = []
        for i, seq in enumerate(sequences):
            trg = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            output = model.decoder(model.embedding(trg), memory)
            logits = model.out(output[:, -1, :])  # Get last token's prediction

            top_k_probs, top_k_ids = torch.topk(torch.softmax(logits, dim=-1), beam_width)
            for k in range(beam_width):
                new_seq = seq + [top_k_ids[0, k].item()]
                new_score = scores[i] + torch.log(top_k_probs[0, k]).item()
                all_candidates.append((new_seq, new_score))

        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        sequences, scores = zip(*all_candidates)

        if all(eos_idx in seq for seq in sequences):
            break

    best_seq = sequences[0]
    return sp.decode(best_seq[1:])  # Remove SOS token

# Chatbot Function
def chatbin(mode="train", qus=None, q_len=None):
    if mode == "train":
        train_model(epochs=5)
    else:
        model.load_state_dict(torch.load("chatbot_model.pth"))
        model.eval()
        response = beam_search(qus, max_len=q_len)
        print("Bot:", response)
