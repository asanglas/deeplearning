import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


'''
The orginal transformer model was initially used to translate sentences from one language to another. We will do it too. Unlike a GPT model, it contains an encoder and decoder part, not just the decoder.

Tutorial: https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil

'''

### ENCODER

'''
In the encoder part of the transformer, we feed first need to encode the sentence. Let's say we take each word in the sentence and assign it an integer number (we tokenize it). Then to each of this numbers we will assing a (learnable) vector of size emb_dim (in the tutorial it is 512, in the paper this is called d_model).
In Pytorch we can use an nn.Embedding module to get this. It's simply a matrix of size vocab_size x emb_size, and it simply selects the row that corresponds to the integer of the word (the total of words is vocab_size) and outputs the vector of size emb_dim 
'''

class TokenEncoding(nn.Module):
    def __init__(self, emb_dim : int, vocab_size : int):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)


    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.emb_dim) # multiplied by sqrt(emb_dim) why? it's in the paper I guess


'''
Aside from the token embedding, we need to tell about the position of the tokens, since the attention mechansim does not tells anything about spatial structures, it is just a communication mechanism. In the paper they use sinusoidal encoding.

context_lenght: the maximum length that we will take as an input
dropout: for training and decrease overfitting...
'''

def sinusoidal_encoding(emb_dim, context_lenght):
    pe = torch.zeros(context_lenght, emb_dim)
    position = torch.arange(0, context_lenght, dtype=torch.float).unsqueeze(1) # the unsqueeze converts it to a (context_length, 1) tensor
    div_term = torch.exp( torch.arange(0, emb_dim, 2, dtype=torch.float) * (- math.log(10000.0)) / emb_dim) # (emb_dim/2)
    # position * div_term = (context_lenght, emb_dim/2)

    pe[:, ::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe # (context_lenght, emb_dim)  

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim : int, context_lenght : int, dropout : float):
       super().__init__() 
       self.emb_dim = emb_dim
       self.context_length = context_lenght
       self.dropout = nn.Dropout(dropout)

       self.register_buffer('positional_encoding', sinusoidal_encoding(self.emb_dim, self.context_length).unsqueeze(0)) # (1, context_length, emb_dim ))


    def forward(self, x):
       # x = (B, text_size, emb_dim)
       x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False) # only the text we have
       return self.dropout(x)
    
class LayerNorm(nn.Module):
    # is basically a batch norm (less complicated) across context dimension
    def __init__(self, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim, hid_dim, dropout : float):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)

        self.ff1 = nn.Linear(emb_dim, hid_dim)
        self.ff2 = nn.Linear(hid_dim, emb_dim)

    
    def forward(self, x):
        x = F.relu(self.ff1(x))
        x = self.dropout(x)
        x = self.ff2(x)
        return x

# the multihead attention dividies the embedding dimension into multiple smaller attentions. 
# the number of heads must divide the input dimension (the context one)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_dim : int, nheads : int, dropout : float):
        super().__init__()
        assert emb_dim % nheads == 0, "nheads must divide emb_dim" # check it is divisible
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.dk = emb_dim // nheads

        self.dropout = nn.Dropout(dropout)
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.ln_out = nn.Linear(emb_dim, emb_dim, bias=False)

   
    @staticmethod
    def attention(query, key, values, mask, dropout: nn.Dropout):
        dk = query.shape[-1]
        attention_scores = query @ key.transpose(-1, -2) / torch.sqrt(dk) # (B, nhead, seq, seq) 
        if mask is not None: # mask if necesary
            attention_scores.masked_fill(mask == 0, -torch.inf)
        attention_scores = attention_scores.softmax(dim=1) # sofmax each row (B, nhead, seq, seq)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        x = attention_scores @ values # (B, nhead, seq, dk) 
        return x, attention_scores 

    def forward(self, qi, ki, vi, mask):
        # x : (B, seq, emb_dim)
        query = self.q(qi)  # (B, seq, emb_dim)
        key = self.k(ki)
        value = self.v(vi)

        # split across the embeding dimension and rearange. We will have (B, nhead, seq, dk)
        query = query.view(query.shape[0],  query.shape[1], self.nheads, self.dk).transpose(1,2) 
        key = key.view(key.shape[0],  key.shape[1], self.nheads, self.dk).transpose(1,2)
        value = value.view(value.shape[0],  value.shape[1], self.nheads, self.dk).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  # (B, nhead, seq, dk) 
        
        #concat
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.nheads * self.dk) # (B, seq, emb_dim). The contiguous is for the use of view?

        x = self.ln_out(x) # (B, seq, emb_dim)

        return x 
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # slight different from the og paper, we first apply the norm, then the multihead block
        return x + self.dropout(self.norm(x))

class EncoderBlock(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, nheads: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(emb_dim, nheads, dropout)
        self.feed_forward = FeedForward(emb_dim, hid_dim)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, nheads: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(emb_dim, nheads, dropout)
        self.cross_attention = MultiHeadAttentionBlock(emb_dim, nheads, dropout) 
        self.feed_forward = FeedForward(emb_dim, hid_dim)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)

        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class LinearLayer(nn.Module): # convert from the embedding to the vocabulary
    def __init__(self, emb_dim : int, vocab_size : int):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.ln = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.ln(x)
        x = F.log_softmax(x, dim=-1) # we return the log softmax, then exponentiate to get probablities, more stable??
        return x  

## THE TRANSFORMER
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

