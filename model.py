# imports
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# ----
# hyperparameters
block_size = 128 
batch_size = 256
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embed = 64 # dimensionality of the character embedding vectors

# ----

# create a model
class BiGramDataModel(nn.Module):
    def __init__(self, vocab_size):
        # initialize with random weights
        super().__init__()
        # warning: generally set to a small value but we have characters for tokens
        # this basically is used to tell you what the probability is of one token
        # coming after another, hence "bigram"
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.feed_forward = FeedForward(n_embed)


    def forward(self, idx, targets=None):
        B, T= idx.shape
        # plugs the tokens into the table 
        # semantic relationship, this is basically the "self-attention" part 
        # of the paper, and these complex weights are the only thing that's
        # really trained here :)
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # add them
        x = token_emb + pos_emb
        # apply the head
        x = self.sa_heads(x)
        x = self.feed_forward(x)

        # apply lm head (linear transformation to return back to life)
        logits = self.lm_head(x)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]

            # get the predictions
            # this calls forward for the tokens 
            logits, loss = self(idx_cond)

            # focus only on the last time step
            # remove T because this is a BiGram model
            logits = logits[:, -1, :]  # get the last time step 

            probs = F.softmax(logits, dim=-1)  # probabilities

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# create a Head
class Head(nn.Module):
    def __init__(self, head_size):
        # initialize with random weights
        super().__init__()
        # add a key, value, and query
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #normalization
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, size):
        super().__init__()
        self.head_list = nn.ModuleList([Head(size) for _ in range(heads)])
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.head_list], dim=-1)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
