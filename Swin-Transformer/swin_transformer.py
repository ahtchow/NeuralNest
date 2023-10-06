#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn

class SwinEmbeddingBlock(nn.Module):
    """
    Swin Embedding: Patch Partition and Linear Embedding
    The efficient method of performing this task is through convolution.

    It first splits an input RGB image into non-overlapping patches by a 
    patch splitting module, like ViT. Each patch is treated as a “token” 
    and its feature is set as a concatenation of the raw pixel RGB values. 
    In our implementation, we use a patch size of 4 × 4 and thus the feature
    dimension of each patch is 4 × 4 × 3 = 48. A linear embedding layer is 
    applied on this raw-valued feature to project it to an arbitrary dimension 
    (denoted as C).”
    
    Reference: Liu Z. et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/pdf/2103.14030.pdf

    """
    def __init__(self, patch_size=4, embed_dim=96):
        super(SwinEmbeddingBlock, self).__init__()

        self.P = patch_size
        self.embed_dim = embed_dim # Refered to as C in paper

        self.swin_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (N, c, H, W)

        Returns:
            Linear Embedding: (N, (H/P)*(W/P), C)
        """
        N, _, H, W = x.shape
        lin_embed = self.swin_embedding(x)

        assert int(H/self.P) == lin_embed.shape[2], "Incorrect shape"
        assert int(W/self.P) == lin_embed.shape[3], "Incorrect shape"

        num_patches = int((H / self.P) * (W / self.P))
        lin_embed = lin_embed.view(N, num_patches, self.embed_dim)
        lin_embed = self.layer_norm(lin_embed)
        return lin_embed


class PatchMergingBlock(nn.Module):
    """
    Patch Merging: 

    To produce a hierarchical representation, the number of tokens is 
    reduced by patch merging layers as the network gets deeper. 
    The first patch merging layer concatenates the features of 
    each group of 2 × 2 neighboring patches, and applies a linear 
    layer on the 4C-dimensional concatenated features. 
    This reduces the number of tokens by a multiple of 2×2 = 4 
    (2× downsampling of resolution), and the output dimension is set to 2C.

    Reference: Liu Z. et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/pdf/2103.14030.pdf

    """
    def __init__(self, embed_dim=96):
        super(PatchMergingBlock, self).__init__()

        self.embed_dim = embed_dim
        self.linear = nn.Linear(4*embed_dim, 2*embed_dim)
        self.layer_norm = nn.LayerNorm(2*embed_dim)

    def forward(self, x):
        """
        Args:
            x: (N, (H x W), C)

        Returns:
            Merged Patches: (N, (H/2 x W/2), 2C)
        """
        N, num_patches, C = x.shape
        assert self.embed_dim == C, "Token dimension is wrong."
        
        # Reorganize into 2D Patches -> (B, H, W, C)
        H = W = int(np.sqrt(num_patches))
        x = x.view(N, H, W, C)

        # Separate per patch by 2 x 2 -> (B, H/2, W/2, C)
        x_top_left = x[:, 0::2, 0::2, :]
        x_bottom_left = x[:, 1::2, 0::2, :]
        x_top_right = x[:, 0::2, 1::2, :]
        x_bottom_right = x[:, 1::2, 1::2, :]

        # Merge by channel -> (B, H/2, W/2, 4C)
        x = torch.cat([x_top_left, x_bottom_left, x_top_right, x_bottom_right], -1)       
        
        # Flatten 2D Patches -> (B, (H/2 x W/2), 4C)
        x = x.view(N, -1, 4 * C)
        
        # Linear and Normalize
        x = self.linear(x)
        x = self.layer_norm(x)

        return x

class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7, shift=False):
        super(ShiftedWindowMSA, self).__init__()
        
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        assert embed_dim % num_heads == 0, f"Can't divide dimension {embed_dim} into {num_heads} heads"
        self.head_dim = int(embed_dim / num_heads)
        self.shift = shift
        
        self.qkv_mapping = nn.Linear(embed_dim, 3*embed_dim)
        self.lin_concat = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Compute shift mask to make sure overflowing tokens do not have
        # attention to tokens that are not originally local.
        self.column_mask = None
        self.row_mask = None
        if self.shift:
            displacement = self.window_size//2
            # Row Mask
            self.row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            self.row_mask[-self.window_size * (displacement):, 0:-self.window_size * (displacement)] = float('-inf')
            self.row_mask[0:-self.window_size * (displacement), -self.window_size * (displacement):] = float('-inf')

            # Column Mask
            self.column_mask = torch.zeros(self.window_size, self.window_size, self.window_size, self.window_size).cuda()
            self.column_mask[:, -displacement, :, :-displacement] = float('-inf')
            self.column_mask[:, :-displacement, :, -displacement:] = float('-inf')
            self.column_mask = self.column_mask.reshape(self.window_size**2, self.window_size**2)

        self.rel_pos_embed_indices, self.rel_pos_embed = self.init_relative_positional_embedding()

    def self_attention(self, Q, K, V):
        '''
        Need dot product across tokens, so swap K's
        num_window_tokens and head_dim dimensionality.
        '''
        attention = (Q @ K.transpose(4, 5)) / (self.head_dim ** 0.5)
        
        # Add relative positional embedding
        B = self.get_relative_positional_embedding()
        attention += B
        
        # Mask the overflowing row and column tokens.
        if self.shift:
            attention[:, :, -1, :] += self.row_mask
            attention[:, :, :, -1] += self.column_mask

        attention = self.softmax(attention) @ V

        return attention
    
    def init_relative_positional_embedding(self):
        # Random Tensor that will be learned.
        rel_pos_embed = nn.Parameter(torch.rand(2*self.window_size-1, 2*self.window_size-1))

        # Compute relative distance for x, y dimensions
        indices = torch.tensor(np.array([[x,y] for x in range(self.window_size) 
                                               for y in range(self.window_size)]))

        rel_distances = indices[None, :, :] - indices[:, None, :] # Distance [-win_size+1,win_size-1]
        rel_pos_embed_indices = rel_distances +  + self.window_size - 1 # Distance [0, 2win_size-2], normalized so we can query pos_embedd. 
        return rel_pos_embed_indices, rel_pos_embed
    
    def get_relative_positional_embedding(self):
        return self.rel_pos_embed[self.rel_pos_embed_indices[:, :, 0], 
                                  self.rel_pos_embed_indices[:, :, 1]] 

    def forward(self, x):
        N, num_patches, C = x.shape
        assert self.embed_dim == C, "Token dimensionality is not correct."
        H = W = int(np.sqrt(num_patches))
        
        # Multiply by linear layer to get Q, K, V based on tokens
        x = self.qkv_mapping(x)
        
        # Reshape into windows -> (N, H, W, C, 3)
        x = x.view(N, H, W, C, 3)

        # Shift windows by floor(window_size/2) in x and y direction
        if self.shift:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        '''
        Reshape into windows (N, num_heads, h, w, num_window_tokens, head_dim, 3)
            N = batch size
            num_heads = # of Attention Heads
            h,w = # of windows vertically and horizontally
            num_window_tokens = total tokens in each window (window_size^2)
            head_dim = head dimension
            3 = a constant to break our matrix into 3 Q,K,V matricies 
        '''
        h = int(H/self.window_size)
        w = int(W/self.window_size)
        num_window_tokens = int(self.window_size**2)
        x = x.view(N, self.num_heads, h, w, num_window_tokens, self.head_dim, 3)
        
        # Perform Self-Attention on Windows/Shifted-Windows
        Q, K, V = x.chunk(3, dim=-1)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        attention = self.self_attention(Q, K, V)

        # Shift windows back into place 
        attention = attention.view(N, H, W, C)
        if self.shift:
            attention = torch.roll(attention, (self.window_size//2, self.window_size//2), (1,2))

        attention = attention.view(N, num_patches, C)
        out = self.lin_concat(attention)
        return out
    
class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mlp_ratio, shift):
        super(SwinEncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift = shift

        # Components
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.SWMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, shift=shift)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*self.mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*self.mlp_ratio, embed_dim)
        )

    def forward(self, x):
        SWMSA = self.SWMSA(self.layer_norm(x))
        SWMSA_residual = SWMSA + x

        MLP = self.dropout(SWMSA_residual)
        MLP = self.layer_norm2(MLP)
        MLP = self.MLP(MLP)
        out = MLP + SWMSA_residual
        return out
    
class AlternatingSwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7 , mlp_ratio=4):
        super(AlternatingSwinEncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.WMSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, 
                                    mlp_ratio=mlp_ratio, shift=False)
        self.SWMSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, 
                                      mlp_ratio=mlp_ratio, shift=True)
    
    def forward(self, x):
        return self.SWMSA(self.WMSA(x))

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(SwinTransformer, self).__init__()
        self.num_classes = num_classes

        # Stage 1
        self.SwinEmbeddingS1 = SwinEmbeddingBlock(patch_size=4, embed_dim=96)
        self.AlternatingSwinEncoderS1 = AlternatingSwinEncoderBlock(embed_dim=96, num_heads=3, window_size=7)

        # Stage 2
        self.PatchMergeS2 = PatchMergingBlock(embed_dim=96)
        self.AlternatingSwinEncoderS2 = AlternatingSwinEncoderBlock(embed_dim=192, num_heads=6, window_size=7)

        # Stage 3
        self.PatchMergeS3 = PatchMergingBlock(embed_dim=192)
        self.AlternatingSwinEncoderS3x1 = AlternatingSwinEncoderBlock(embed_dim=384, num_heads=12, window_size=7)
        self.AlternatingSwinEncoderS3x2 = AlternatingSwinEncoderBlock(embed_dim=384, num_heads=12, window_size=7)
        self.AlternatingSwinEncoderS3x3 = AlternatingSwinEncoderBlock(embed_dim=384, num_heads=12, window_size=7)

        # Stage 4
        self.PatchMergeS4 = PatchMergingBlock(embed_dim=384)
        self.AlternatingSwinEncoderS4 = AlternatingSwinEncoderBlock(embed_dim=768, num_heads=24, window_size=7)

        # Classification Head
        self.layer_norm = nn.LayerNorm(768)
        self.GELU = nn.GELU()
        self.Dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(768, self.num_classes),
            nn.Softmax(dim=-1)
        )

        # Initialize Weights
        self.apply(self.init_weights)
    
    def forward(self, x):
        
        # Stage 1
        x = self.SwinEmbeddingS1(x) #(N, H/4 * W/4, 96)
        x = self.AlternatingSwinEncoderS1(x) #(N, H/4 * W/4, 96)

        # Stage 2
        x = self.PatchMergeS2(x) #(N, H/8 * W/8, 192)
        x = self.AlternatingSwinEncoderS2(x) #(N, H/8 * W/8, 192)

        # Stage 3
        x = self.PatchMergeS3(x) #(N, H/16 * W/16, 384)
        x = self.AlternatingSwinEncoderS3x1(x) #(N, H/16 * W/16, 384)
        x = self.AlternatingSwinEncoderS3x2(x) #(N, H/16 * W/16, 384)
        x = self.AlternatingSwinEncoderS3x3(x) #(N, H/16 * W/16, 384)

        # Stage 4
        x = self.PatchMergeS4(x) #(N, H/32 * W/32, 768)
        x = self.AlternatingSwinEncoderS4(x) #(N, H/32 * W/32, 768)

        # Classification Head
        x = self.layer_norm(self.GELU(x))
        x = self.avg_pool(x.transpose(1,2))
        x = torch.flatten(x, 1)
        x = self.Dropout(x)
        out = self.cls_head(x)
        return out
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, 0.0, 0.02)
            m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
