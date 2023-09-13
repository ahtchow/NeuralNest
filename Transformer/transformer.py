#!/usr/bin/env python3
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocb_size, emb_dim, max_seq_len, num_layers, num_head, ffn_hidden_dim, dropout_prob=0.1):
        super(Transformer, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.target_vocb_size = target_vocb_size
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_head = num_head
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout_prob = dropout_prob

        # layers
        self.encoder = Encoder(input_vocab_size,
                               emb_dim,
                               max_seq_len,
                               num_layers,
                               num_head,
                               ffn_hidden_dim,
                               dropout_prob)

        self.decoder = Decoder(input_vocab_size,
                               emb_dim,
                               max_seq_len,
                               num_layers,
                               num_head,
                               ffn_hidden_dim,
                               dropout_prob)
        
        self.linear = nn.Linear(emb_dim, target_vocb_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp, target, enc_mask=None, dec_target_mask=None, dec_src_mask=None):
        enc_out = self.encoder(inp, enc_mask)
        dec_out = self.decoder(target, enc_out, dec_target_mask, dec_src_mask)
        ff_out = self.linear(dec_out)
        out = self.softmax(ff_out)
        return out
