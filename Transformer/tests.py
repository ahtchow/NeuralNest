#!/usr/bin/env python3
import torch

def test_positional_encoding_block(PositionalEncoding, max_seq_len, emb_dim):
    max_seq_len = torch.tensor(max_seq_len)
    emb_dim = torch.tensor(emb_dim)

    pos_encoding = PositionalEncoding(max_seq_len)
    sin_part = pos_encoding[:, 0::2]
    cos_part = pos_encoding[:, 1::2]

    assert torch.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (max_seq_len, emb_dim), f"Wrong shape. We expected: ({max_seq_len}, {emb_dim})"

    ones = sin_part ** 2  +  cos_part ** 2
    assert torch.allclose(ones, torch.ones((max_seq_len, emb_dim // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
    
    angs = torch.arctan(sin_part / cos_part)
    angs[angs < 0] += torch.pi
    angs[sin_part < 0] += torch.pi
    angs = angs % (2 * torch.pi)
    
    pos_m = torch.arange(max_seq_len)[:, None]
    dims = torch.arange(emb_dim)[None, :]

    angles = pos_m/ (torch.pow(10000, (2 * (dims//2)) / emb_dim.float()))
    trueAngs = angles[:, 0::2] % (2 * torch.pi)
    
    assert torch.allclose(angs, trueAngs), "Did you apply sin and cos to even and odd parts respectively?"
 
    print("\033[92mPositional-Encoding Block: All tests passed")


def test_self_attention_block(SelfAttention):
    q = torch.tensor([[[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]], dtype=torch.float32)
    k = torch.tensor([[[[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1]]]], dtype=torch.float32)
    v = torch.tensor([[[[0, 0], [1, 0], [1, 0], [1, 1]]]], dtype=torch.float32)

    attention, weights = SelfAttention(q, k, v, None)
    assert torch.is_tensor(weights), "Weights must be a tensor"
    assert tuple(weights.size()[2:]) == (q.shape[2], k.shape[3]), f"Wrong shape. We expected ({q.shape[2]}, {k.shape[3]})"
    assert torch.allclose(weights, torch.tensor([[[[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]]]]))

    assert torch.is_tensor(attention), "Output must be a tensor"
    assert tuple(attention.size()[2:]) == (q.shape[2], v.shape[3]), f"Wrong shape. We expected ({q.shape[2]}, {v.shape[3]})"
    assert torch.allclose(attention, torch.tensor([[[[0.74105227, 0.15705977],
                                [0.7227253,  0.16817567],
                                [0.6637989,  0.2039163 ]]]]))

    mask = torch.tensor([[[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]])
    attention, weights = SelfAttention(q, k, v, mask)
    assert torch.allclose(weights, torch.tensor([[[[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]]])), "Wrong masked weights"
    assert torch.allclose(attention, torch.tensor([[[[0.6928040981292725, 0.18632373213768005],
                                [0.6163482666015625, 0.2326965481042862], 
                                [0.6163482666015625, 0.2326965481042862]]]])), "Wrong masked attention"
    
    print("\033[92mSelf-Attention Block: All tests passed")

def test_multi_head_attention_block(MultiHeadAttention, emb_dim, num_heads):
    mha_torch = torch.nn.MultiheadAttention(emb_dim, num_heads)
    W_q, W_k, W_v = mha_torch.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha_torch.in_proj_bias.chunk(3, dim=0)
    concat_weight = mha_torch.out_proj.weight
    concat_bias = mha_torch.out_proj.bias

    # Set weights to be the same
    MultiHeadAttention.W_q.weight.data = W_q
    MultiHeadAttention.W_k.weight.data = W_k
    MultiHeadAttention.W_v.weight.data = W_v
    MultiHeadAttention.W_concat.weight.data = concat_weight
    MultiHeadAttention.W_q.bias.data = b_q
    MultiHeadAttention.W_k.bias.data = b_k
    MultiHeadAttention.W_v.bias.data = b_v
    MultiHeadAttention.W_concat.bias.data = concat_bias

    batch_size = 8
    seq_len = 10
    src = torch.rand(batch_size, seq_len, 3 * emb_dim)
    queries, keys, values = src.chunk(3, dim=-1)
    attention = MultiHeadAttention(queries, keys, values)
    torch_attention, _ = mha_torch(queries.permute(1, 0, 2), keys.permute(1, 0, 2), values.permute(1, 0, 2))
    assert torch.allclose(attention, torch_attention.permute(1, 0, 2)), "Wrong multi-headed attention"

    print("\033[92mMulti-Head Attention Block: All tests passed")

def test_layer_norm_block(LayerNorm, emb_dim):
    ln_torch = torch.nn.LayerNorm(emb_dim)
    gamma = ln_torch.weight.data
    beta = ln_torch.bias.data
    eps = ln_torch.eps

    # Set params to be the same
    LayerNorm.gamma.data = gamma
    LayerNorm.beta.data = beta
    LayerNorm.eps = eps

    batch_size = 8
    seq_len = 10
    src = torch.rand(batch_size, seq_len, emb_dim)
    src_normalized = LayerNorm(src)
    src_normalized_torch = ln_torch(src)
    assert torch.allclose(src_normalized, src_normalized_torch, atol=1e-06), "Wrong layer normalization implementation"

    print("\033[92mLayer-Norm Block: All tests passed")
