# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenization
data = ""
with open("data.txt", "r") as f:
    data = f.read()

# tokens 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique" % (data_size, vocab_size))

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

# warning: susceptible to overlapping data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i+block_size+1] for i in ix])
    return x, y

# create a model
class BiGramDataModel(nn.Module):
    def __init__(self, vocab_size):
        # initialize with random weights
        super().__init__()
        # warning: generally set to a small value but we have characters for tokens
        # this basically is used to tell you what the probability is of one token
        # coming after another, hence "bigram"
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        # if there's nothing we compare to
        if targets is None:
            loss = None
        else:
            # batch size, sequence length, classes/dimensions
            # number of sequences being processed simultaneously WHY DO WE CARE
            # number of time steps/token per sequence (block_size)
            # contains informatation about the tokens before it, the \
            # "density" of each token
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


