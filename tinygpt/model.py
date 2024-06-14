import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ------------------------------
# 
# Token Embedding
#
# ------------------------------

# For each token in the sequnce (a number within the range of vocab_size) will be assign a vector of size n_embed fom the embedding table.
# This encodes the identity of each token

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size : int, n_embed : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.embedding = nn.Embedding(vocab_size, n_embed)

    def forward(self, x):
        # multiplied by sqrt(d_model) in the paper
        return self.embedding(x) * np.sqrt(self.n_embed)  # (batch, block_size) -> (batch, block_size, n_embed)

# ------------------------------
# 
# Positional Embedding
#
# ------------------------------

# We also encode the position

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size : int, n_embed : int):
        super().__init__()
        self.context_length = block_size
        self.n_embed = n_embed
        self.embedding = nn.Embedding(block_size, n_embed)

    def forward(self, x):
        # Here the input will be a tensor of the form (0, 1, 2, ... block_size)
        return self.embedding(x) * np.sqrt(self.n_embed)  # (block_size) -> (block_size, n_embed)
    

# ------------------------------
# 
# Projection Layer
#
# ------------------------------

#  This is just a linear layer that projects from embedding space to vocab space. It is applied at the end of the transformer

class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size : int, n_embed : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.projection_layer = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        return self.projection_layer(x)   # (batch, block_size, n_embed) -> (batch, block_size, vocab_size)
    


# ------------------------------
# 
# Muli-Head Attention
#
# ------------------------------

# 1. The multihead attention dividies the embedding dimension into multiple smaller attentions. 
# 2. The number of heads must divide the input dimension (the sequence one)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, n_embed : int, num_heads : int, block_size : int , dropout : float):
        super().__init__()
        assert n_embed % num_heads == 0, "num_heads must divide n_embed" # check that n_embed can be divided by the num of heads
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.query = nn.Linear(n_embed, n_embed, bias=False) # Wq
        self.key = nn.Linear(n_embed, n_embed, bias=False) # Wk
        self.value = nn.Linear(n_embed, n_embed, bias=False) # Wv
        self.proj = nn.Linear(n_embed, n_embed, bias=False) # Wo

        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size))) # (block_size, block_size) lower diagonal matrix
   
        self.dropout = None if dropout == None else nn.Dropout(dropout)
        self.attention_scores = None


    def forward(self, x):
        batch, x_block_size, n_emb = x.shape
        
        q = self.query(x) # (batch, x_block_size, n_emb) @ (batch, n_emb, n_emb) -> (batch, x_block_size, n_emb)
        k = self.key(x) 
        v = self.value(x) 

        # Split across the embedding dimension and rearange.
        # (batch, x_block_size, n_emb) -> (batch, x_block_size, num_head, head_size) ->  (batch, num_head, x_block_size, head_size)
        q = q.view(q.shape[0],  q.shape[1], self.num_heads, self.head_size).transpose(1,2) # (batch, num_head, x_block_size, head_size)
        k = k.view(k.shape[0],  k.shape[1], self.num_heads, self.head_size).transpose(1,2)
        v = v.view(v.shape[0],  v.shape[1], self.num_heads, self.head_size).transpose(1,2)

        # Get the weights
        weights = q @ k.transpose(-1, -2) / np.sqrt(self.head_size) # (batch, num_head, x_block_size, head_size) @  (batch, num_head, head_size, x_block_size) -> (batch, num_head, x_block_size, x_block_size)

        # Causal mask 
        weights = weights.masked_fill(self.causal_mask[:x_block_size, :x_block_size] == 0, float('-inf')) # (batch, num_head, x_block_size, x_block_size)

        # Apply softmax (the -inf will be 0)
        weights = torch.softmax(weights, dim=-1) # sofmax each row (batch, num_heads, x_block_size, x_block_size)
        
        # Apply dropout
        # if self.dropout is not None: weights = self.dropout(weights)
        
        # Finally multiply by the values
        out =  weights @ v # (batch, num_head, x_block_size, x_block_size) @ (batch, num_head, x_block_size, head_size) -> (batch, num_head, x_block_size, head_size)
        
        # save the weights
        self.attention_scores = weights
        
        # Concat the heads
        # (batch, num_heads, x_block_size, head_size) -> (batch, x_block_size, num_heads, head_size) -> (batch, x_block_size, n_embed)
        out = out.transpose(1,2).contiguous().view(out.shape[0], -1, self.num_heads * self.head_size) # (batch, x_block_size, n_embed). 
        
        out = self.proj(out) # (batch, x_block_size, n_embed) @ (n_embed, n_embed) -> (batch, x_block_size, n_embed)

        # Apply dropout
        if self.dropout is not None: out = self.dropout(out)
        
        return out


# ------------------------------
# 
# Feed Forward
#
# ------------------------------

class FeedForward(nn.Module):
    def __init__(self, n_embed : int,  dropout : float):
        super().__init__()
        self.n_embed = n_embed
        self.ff1 = nn.Linear(n_embed, 4*n_embed) # hard code the hiden dim
        self.ff2 = nn.Linear(4*n_embed, n_embed)
        
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x):
        x = F.relu(self.ff1(x)) # (batch, x_block_size, n_embed) -> (batch, x_block_size, 4 * n_embed)
        x = self.ff2(x) # (batch, x_block_size, 4*n_embed) -> (batch, x_block_size, n_embed)
        x = self.dropout(x)
        return x
    

# ------------------------------
# 
# Layer Norm
#
# ------------------------------

# Check the notes
# TODO: check the mean(dim=-1). The sequence dimension is the dim = 1, no?
# Normalized the rows instead of the columns (batch norm). This is taken from Karpathy, I have to check exaclty

class LayerNorm(nn.Module):

  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def forward(self, x):
    # calculate the forward pass
    xmean = x.mean(-1, keepdim=True) # batch mean
    xvar = x.var(-1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  

# ------------------------------
# 
# Transformer Layer
#
# ------------------------------


class Layer(nn.Module):
    def __init__(self, n_embed: int, num_heads: int, block_size : int,  dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(n_embed=n_embed, num_heads=num_heads, block_size=block_size, dropout=dropout)
        self.feed_forward = FeedForward(n_embed=n_embed, dropout=dropout)
        self.layer_norm1 = LayerNorm(n_embed)
        self.layer_norm2 = LayerNorm(n_embed)


    def forward(self, x):
        # Add the residual connections + layer norms
        x = x + self.self_attention(self.layer_norm1(x)) # (batch, x_block_size, n_embed) ->  (batch, x_block_size, n_embed)
        x = x + self.feed_forward(self.layer_norm2(x)) # (batch, x_block_size, n_embed) ->  (batch, x_block_size, n_embed)
        return x
    

# ------------------------------
# 
# The Transformer
#
# ------------------------------


class Transformer(nn.Module):
    def __init__(self, vocab_size : int, n_embed : int, num_heads: int, num_layers: int,  block_size : int, dropout : float):
        super().__init__()
        self.vocab_size = vocab_size 
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.dropout = dropout 

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, n_embed=n_embed)
        self.positional_embedding = PositionalEmbedding(block_size=block_size, n_embed=n_embed)

        # attention
        # self.sa_block = MultiHeadAttentionBlock(n_embed=n_embed, num_heads=num_heads, block_size=block_size, dropout=None)
        # self.ff_block = FeedForward(n_embed=n_embed, hid_dim=4*n_embed, dropout=0.0)

        self.layers = nn.Sequential( 
            *[ Layer(n_embed=n_embed, num_heads=num_heads, block_size=block_size, dropout=dropout) for _ in range(num_layers)] )


        self.projection_layer = ProjectionLayer(vocab_size=vocab_size, n_embed=n_embed)

        # Initialize the parameters
        self._init_parameters()

    def forward(self, idx, targets=None):
        batch_size, idx_block_size = idx.shape

        device = idx.device 

        tok_emb = self.token_embedding(idx) # (batch, block_size) -> (batch, block_size, n_embed)
        pos_emb = self.positional_embedding(torch.arange(idx_block_size, device=device)) # (block_size, n_embed)
        x = tok_emb + pos_emb # broadcasting works. (batch, block_size, n_embed)

        # attention
        x = self.layers(x) # (batch, idx_block_size, n_embed) -> (batch, idx_block_size, n_embed)


        # the last layer
        logits = self.projection_layer(x) # (batch, block_size, n_embed) -> (batch, block_size, vocab_size)

        loss = None
        if targets != None: 
            # cross entropy expects the form (batch, classes). Essentially, each token in a sequence acts as an independent input. So we can group it with the batch 
            logits = logits.view(logits.shape[0] * idx_block_size, self.vocab_size) # (batch, block_size, vocab_size) -> (batch * block_size, vocab_size) 
            # The targets are of the form (batch, block_size)
            targets = targets.view(targets.shape[0] * idx_block_size) # no need to create a one hot encoding. In this form, cross entropy does it for you (CHECK)
            loss = F.cross_entropy(logits, targets) 
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop the idx so that only take the maximum block size.
            idx_crop = idx[:, -self.block_size:]
            # forward pass the model
            logits, _ = self.forward(idx_crop) # (batch = 1 for 1 prediciton, block_size) -> (batch, block_size, vocab_size)
            # get the last token, we want to predict the next token from the last (given all the context, etc...)
            logits = logits[:, -1, :] # (batch, block_size, vocab_size) -> (batch, vocab_size)
            # apply softmax to get a probability distribution across the vocabulary
            probs = F.softmax(logits, dim=-1) 

            # Then we could get the token with most associated probability for the next token. But we can also sample from this
            idx_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
            
            # Append the token to the previous one, that serves as a new context
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

    # Initiliazion of the parameters, as in the paper. I think PyTorch does it automatically, because all are linear layers. 
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

