#!/usr/bin/env python3
import torch
from model.ViT import ViT

def test_patchify():
    # Test 1
    config = {'H': 28, 'W': 28, 'C': 1, 'P': 7, 'LP': 8,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 5, 1, 28, 28
    patch_dim = config['P']
    x = torch.randn(n, c, h, w)
    out = model.patchify(x)

    target_shape = (n, patch_dim**2, (h*c)/patch_dim * (w*c)/patch_dim)
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape,  f"Wrong shape. Got {out.shape}. We expected: {target_shape}"
    
    # Test 2
    config = {'H': 112, 'W': 112, 'C': 3, 'P': 7, 'LP': 420,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 10, 3, 112, 112
    patch_dim = config['P']
    x = torch.randn(n, c, h, w)
    out = model.patchify(x)
    target_shape = (n, patch_dim**2, int(h/patch_dim * w/patch_dim * c))
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape, f"Wrong shape. Got {out.shape}. We expected: {target_shape}"

    print("\033[92mPatchification: All tests passed")

def test_linear_projection():
    # Test 1
    config = {'H': 28, 'W': 28, 'C': 1, 'P': 7, 'LP': 8,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 5, 1, 28, 28
    patch_dim = config['P']
    x = torch.randn(n, patch_dim**2, int(h/patch_dim * w/patch_dim * c))
    out = model.linear_projection(x)
    target_shape = (n, patch_dim**2, config["LP"])
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape,  f"Wrong shape. Got {out.shape}. We expected: {target_shape}"

    # Test 2
    config = {'H': 112, 'W': 112, 'C': 3, 'P': 7, 'LP': 420,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 10, 3, 112, 112
    patch_dim = config['P']
    x = torch.randn(n, patch_dim**2, int(h/patch_dim * w/patch_dim * c))
    out = model.linear_projection(x)
    target_shape = (n, patch_dim**2, config["LP"])
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape, f"Wrong shape. Got {out.shape}. We expected: {target_shape}"

    print("\033[92mLinear Projection: All tests passed")

def test_add_class_embedding():
    # Test 1
    config = {'H': 28, 'W': 28, 'C': 1, 'P': 7, 'LP': 8,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 5, 1, 28, 28
    patch_dim = config['P']
    x = torch.randn(n, patch_dim**2, config["LP"])
    out = model.add_class_embedding(x)
    target_shape = (n, patch_dim**2 + 1 , config["LP"])
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape,  f"Wrong shape. Got {out.shape}. We expected: {target_shape}"

    # Test 2
    config = {'H': 112, 'W': 112, 'C': 3, 'P': 7, 'LP': 420,
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}
    model = ViT(config)
    n, c, h, w = 10, 3, 112, 112
    patch_dim = config['P']
    x = torch.randn(n, patch_dim**2, config["LP"])
    out = model.add_class_embedding(x)
    target_shape = (n, patch_dim**2 + 1, config["LP"])
    assert torch.is_tensor(out), "Output is not a tensor"
    assert out.shape == target_shape, f"Wrong shape. Got {out.shape}. We expected: {target_shape}"

    print("\033[92mAdd Class: All tests passed")