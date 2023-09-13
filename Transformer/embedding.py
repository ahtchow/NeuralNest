#!/usr/bin/env python3
import torch
import torch.nn as nn
from layers import PositionalEncoding

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, emb_dim):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, emb_dim, padding_idx=1)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, emb_dim, max_seq_len, drop_prob):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.tok_emb = TokenEmbedding(vocab_size, emb_dim)
        self.pos_emb = PositionalEncoding(max_seq_len, emb_dim)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x.size()[-1])
        return self.drop_out(tok_emb + pos_emb)

