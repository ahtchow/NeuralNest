#!/usr/bin/env python3
import torch
import torch.nn as nn

from layers import MultiHeadAttention, LayerNorm, FeedForward
from embedding import TransformerEmbedding

class DecoderLayer(nn.Module):
    
    def __init__(self, emb_dim, num_head, ffn_hidden_dim, dropout_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_head = num_head
        self.dropout_prob = dropout_prob

        # Layers
        self.self_attention = MultiHeadAttention(emb_dim, num_head)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.layernorm1 = LayerNorm(emb_dim)
        self.enc_dec_attention = MultiHeadAttention(emb_dim, num_head)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.layernorm2 = LayerNorm(emb_dim)
        self.feedforward = FeedForward(emb_dim, ffn_hidden_dim, dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.layernorm3 = LayerNorm(emb_dim)

    def forward(self, dec_in, enc_out, target_mask, src_mask):
        # 1. Multi-head attention (self)
        self_mha_out = self.self_attention(q=dec_in, k=dec_in, v=dec_in, mask=target_mask)
        self_mha_out = self.dropout1(self_mha_out)
        ff_in = self.layernorm1(self_mha_out + dec_in)
        # 2. Multi-head attention (with encoder out)
        if enc_out is not None:
            ff_in_prior = ff_in
            ff_in = self.enc_dec_attention(q=dec_in, k=enc_out, v=enc_out, mask=src_mask)
            ff_in = self.dropout2(ff_in)
            ff_in = self.layernorm2(ff_in + ff_in_prior)
        # 3. Feed forward
        ffn_out = self.feedforward(ff_in)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(ffn_out + ff_in)
        return out

class Decoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, max_seq_len, num_layers, num_head, ffn_hidden_dim, dropout_prob=0.1):
        super(Decoder, self).__init__()
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
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(emb_dim, 
                                                  num_head, 
                                                  ffn_hidden_dim, 
                                                  dropout_prob)
                                            for _ in range(num_layers)])
        
    def forward(self, dec_in, enc_out, target_mask, src_mask):
        dec_in = self.emb(dec_in)
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(dec_in, enc_out, target_mask, src_mask)
        return out
