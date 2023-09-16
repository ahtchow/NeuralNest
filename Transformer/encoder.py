#!/usr/bin/env python3
import torch
import torch.nn as nn

from layers import MultiheadAttention, LayerNorm, FeedForward
from embedding import TransformerEmbedding

class EncoderLayer(nn.Module):
    
    def __init__(self, emb_dim, num_head, ffn_hidden_dim, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_head = num_head
        self.dropout_prob = dropout_prob

        # Layers
        self.attention = MultiheadAttention(emb_dim, num_head)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.layernorm1 = LayerNorm(emb_dim)
        self.feedforward = FeedForward(emb_dim, ffn_hidden_dim, dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.layernorm2 = LayerNorm(emb_dim)

    def forward(self, x, mask):
        # 1. Multi-head attention 
        mha_out = self.attention(q=x, k=x, v=x, mask=mask)
        # 2. Add & Norm
        mha_out = self.dropout1(mha_out)
        add_norm_out = self.layernorm1(mha_out + x)
        # 3. Feedfoward
        ffn_out = self.feedforward(add_norm_out)
        # Add & Norm
        ffn_out = self.dropout2(ffn_out)
        out = self.layernorm2(ffn_out + add_norm_out)
        return out

class Encoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, max_seq_len, num_layers, num_head, ffn_hidden_dim, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_head = num_head
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout_prob = dropout_prob

        self.emb = TransformerEmbedding(vocab_size, 
                                        emb_dim, 
                                        max_seq_len, 
                                        dropout_prob)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(emb_dim, 
                                                  num_head, 
                                                  ffn_hidden_dim, 
                                                  dropout_prob)
                                            for _ in range(num_layers)])
        
    def forward(self, x, mask):
        x = self.emb(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x
