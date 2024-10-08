# imports
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BiGramDataModel

# ----
# hyperparameters

block_size = 8 
batch_size = 8
max_iters = 10000
eval_interval = 500
eval_iters = 200
device = torch.device("mps")
# ----

# warning: susceptible to overlapping data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# tokenization
data = ""
with open("data.txt", "r") as f:
    data = f.read()

# tokens 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_size, vocab_size))

m = BiGramDataModel(vocab_size)
m = m.to(device)

# mapping
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

## encode and decode
encode = lambda x: [char_to_ix[c] for c in x]
decode = lambda x: "".join([ix_to_char[i] for i in x])

# convert the data to numbers
data_num = torch.tensor(encode(data), dtype=torch.long).to(device)

## decide training and validation data
n = int(0.9 * len(data_num))
train_data = data_num[:n]
val_data = data_num[n:]

x = train_data[:block_size]  # input
y = train_data[1:block_size + 1]  # labels


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

logits, loss = m(xb, yb)
print(logits.shape, loss)

# print untrained output for testing
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

# optimize 

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


for steps in range(max_iters):
    # hangs over here somewhere no idea why

    # every once in a while evaluate the loss on train and val sets
    if steps % eval_interval == 0 or steps == max_iters - 1:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)

    # approach the optimized gradient
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # print somewhat trained data

# pickle the results
torch.save((vocab_size, char_to_ix, ix_to_char, m.state_dict()), "model.pkl")
