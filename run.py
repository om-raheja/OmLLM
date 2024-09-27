import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BiGramDataModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In run.py
vocab_size, char_to_ix, ix_to_char, state = torch.load("model.pkl", weights_only=False)
model = BiGramDataModel(vocab_size)  # Initialize your model first
encode = lambda x: [char_to_ix[c] for c in x]
decode = lambda x: "".join([ix_to_char[i] for i in x])
model.load_state_dict(state)
model.to(device)  # Move to the appropriate device

# generate
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
