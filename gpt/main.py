import torch
from torch import nn
from torch.nn import functional as F

BLOCK_SIZE=32
BATCH_SIZE = 32
EVAL_FREQ = 200
NUM_STEP = 10**5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_text():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_batch(split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])

    return x.to(device), y.to(device)

def encoder_decoder(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[c] for c in l])
    return encode, decode

@torch.no_grad()
def eval(model):
    model.eval()
    xtrain, ytrain = get_batch('train', BLOCK_SIZE, BATCH_SIZE)
    xval, yval = get_batch('val', BLOCK_SIZE, BATCH_SIZE)
    _, train_loss = model.forward(xtrain, ytrain)
    _, val_loss = model.forward(xval, yval)
    print ("Train Loss: {:.4f}".format(train_loss), "Val Loss: {:.4f}".format(val_loss))
    model.train()



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx, None)
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, 1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx



if __name__ == '__main__':
    text = read_text()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encode, decode = encoder_decoder(chars)

    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]

    model = BigramLanguageModel(vocab_size)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(NUM_STEP):
        xb, yb = get_batch('train', BLOCK_SIZE, BATCH_SIZE)
        logits, loss = model(xb, yb)
        optim.zero_grad()
        loss.backward()

        optim.step()
        if (epoch+1) % EVAL_FREQ == 0:
            print (f"Step [{epoch + 1} /{NUM_STEP}]")
            eval(model)

    print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
