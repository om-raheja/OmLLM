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
block_size = 32
batch_size = 32
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embed = 64 # dimensionality of the character embedding vectors

# ----

# warning: susceptible to overlapping data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i+block_size+1] for i in ix])
    return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenization
data = ""
with open("data.txt", "r") as f:
    data = f.read()

# tokens 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_size, vocab_size))

if __name__ == "__main__":
    # mapping
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    ## encode and decode
    ## warning: there is a way with vectors that 3Blue1Brown explains it detail but 
    ## this is basic
    encode = lambda x: [char_to_ix[c] for c in x]
    decode = lambda x: "".join([ix_to_char[i] for i in x])

    # convert the data to numbers
    data_num = torch.tensor(encode(data), dtype=torch.long)

    ## decide training and validation data
    n = int(0.9 * len(data_num))
    train_data = data_num[:n]
    val_data = data_num[n:]

    batch_size = 4
    block_size = 8  # context length: how many characters do we take to predict the next one?

    x = train_data[:block_size]  # input
    y = train_data[1:block_size + 1]  # labels


    xb, yb = get_batch('train')
    print('inputs:')
    print(xb.shape)
    print(xb)
    print('targets:')
    print(yb.shape)
    print(yb)

    m = BiGramDataModel(vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape, loss)
    m = m.to(device)

    # print untrained output for testing
    print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    # optimize 

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


    # it hovers around 2.3-2.8 loss, needs some internal optimization
    for steps in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)

        # approach the optimized gradient
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(steps, loss.item())

    # print somewhat trained data
    print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


    # pickle the results
    torch.save((vocab_size, char_to_ix, ix_to_char, m.state_dict()), "model.pkl")
