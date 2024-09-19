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
        # plugs the tokens into the table 
        # semantic relationship, this is basically the "self-attention" part 
        # of the paper, and these complex weights are the only thing that's
        # really trained here :)
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

            # view it as a 2D tensor of what all has been processed 
            # that way we can have entropy
            logits = logits.view(B * T, C)

            # targets is the same thing except there is no output size
            # they don't care about storing context of each token in 
            # the output
            targets = targets.view(B * T)

            # quantifying the information encoded between what it is
            # and what the semantic relationship should identify
            # warning: I don't really understand what this does
            # also can the same text have different semantic relationship?
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            # this calls forward for the tokens 
            logits, loss = self(idx)

            # focus only on the last time step
            # remove T because this is a BiGram model
            logits = logits[:, -1, :]  # get the last time step 

            probs = F.softmax(logits, dim=-1)  # probabilities

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

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

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

