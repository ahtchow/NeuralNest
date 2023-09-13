#!/usr/bin/env python3
import torch
import torch.nn as nn

import tests

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.encoding = torch.zeros(max_seq_len, emb_dim)

        # Computes the positional embedding for max sequence length
        # forward() returns position embedding for a given sequence's length
        pos = torch.arange(0, self.max_seq_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, self.emb_dim, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.emb_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.emb_dim)))

    def forward(self, seq_len):
        assert (seq_len <= self.max_seq_len), "Sequence length exceeded maximum sequence length"
        return self.encoding[:seq_len, :]


class SelfAttention(nn.Module):

    # Also refered to as Scaled Dot-Product Attention in Paper

    def __init__(self):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):

        batch_size, num_heads, seq_len, emb_dim = keys.size()
        d_k = torch.tensor(emb_dim)

        # 1. Dot Product between Queries and Keys
        score = torch.matmul(queries, keys.transpose(2, 3)) / torch.sqrt(d_k)

        # 2. Apply Masking (Optionally)
        if mask is not None:
            # Set masks to -ve inf so that softmax do not affect score
            score += (mask * -1e9)

        # 3. Apply Softmax
        weights = self.softmax(score)
        
        # 4. Compute Attention(Q, K, V), multiply with values
        attention = torch.matmul(weights, values)

        return attention, weights

class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.emb_dim = emb_dim
        self.attention = SelfAttention()
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_concat = nn.Linear(emb_dim, emb_dim)

    def split(self, tensor):
        """
        Increased Capacity without Increasing Computational Complexity.
        Split embedding by number of heads. So, you get # of heads times the 
        attention patterns with the same computation as a single set 
        of weights operating on the full embedding dimensions.

        :param tensor: [batch_size, seq_len, emb_dim]
        :return: [batch_size, head, seq_len, split_dim]
        """
        batch_size, seq_len, emb_dim = tensor.size()
        assert emb_dim % self.num_head == 0, "The number of heads given is not divisible into the embedding dimension"
        split_dim = emb_dim // self.num_head
        # transpose dim=1, dim=2 since it is easier to perform operations
        return tensor.view(batch_size, seq_len, self.num_head, split_dim).transpose(1, 2)

    def concat(self, tensor):
        """
        Concatenation operation used to combine all the attention heads.

        :param tensor: [batch_size, head, seq_len, split_dim]
        :return: [batch_size, seq_len, emb_dim]
        """
        batch_size, num_head, seq_len, split_dim = tensor.size()
        emb_dim = num_head * split_dim
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)

    def forward(self, queries, keys, values, mask=None):
        queries, keys, values = self.W_q(queries), self.W_k(keys), self.W_v(values)
        queries, keys, values = self.split(queries), self.split(keys), self.split(values)
        attention, weights = self.attention(queries, keys, values, mask=mask)
        concat_attention = self.concat(attention)
        multi_attention = self.W_concat(concat_attention)
        return multi_attention

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))       
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = self.gamma * x_norm + self.beta
        return x_norm

def main():
    tests.test_positional_encoding_block(PositionalEncoding(8, 16), 8, 16)
    tests.test_self_attention_block(SelfAttention())
    tests.test_multi_head_attention_block(MultiHeadAttention(512, 8), 512, 8)
    tests.test_layer_norm_block(LayerNorm(512), 512)

if __name__ == "__main__":
    main()