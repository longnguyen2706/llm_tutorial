import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import read_text, encoder_decoder, get_batch, get_train_val, get_device, eval_model

BLOCK_SIZE = 32  # len of the context windows
BATCH_SIZE = 32
EVAL_FREQ = 200
NUM_STEP = 10 ** 5


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        # train the embedding to predict next word as a classification problem
        # aka, how much loss compare to the next token idx
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx, None)
            logits = logits[:, -1, :]  # B, C -> only take the last idx output
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, 1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx

text = read_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)

encode, decode = encoder_decoder(chars)
device = get_device()
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data, val_data = get_train_val(data)

model = BigramLanguageModel(vocab_size)
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(NUM_STEP):
    xb, yb = get_batch('train', train_data, val_data, BLOCK_SIZE, BATCH_SIZE)
    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()

    optim.step()
    if (epoch + 1) % EVAL_FREQ == 0:
        print(f"Step[{epoch + 1} / {NUM_STEP}]")
        eval_model(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE)

print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
