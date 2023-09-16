import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

class ViT(nn.Module):

    def __init__(self, config, verbose=False):
        super(ViT, self).__init__()
        self.config = config
        self.verbose = verbose

        # Dimensions of Images
        self.H = config['H']
        self.W = config['W']
        self.C = config['C']

        if self.verbose:
            print("\nVision Transformer Configuration: ")
            print(" Input Dimensions (HWC): ({},{},{})".format(self.H, self.W, self.C))

        # 1. Patchification
        self.patch_dim = config['P'] # number of patches along one dimension
        assert self.H % self.patch_dim == 0, "Height of Input must be divisible by patch resolution (P)" 
        assert self.W % self.patch_dim == 0, "Width of Input not entirely divisible by patch resolution (P)" 
        self.patch_H = (self.H) / self.patch_dim
        self.patch_W = (self.W) / self.patch_dim
        self.patch_flat = int(self.patch_H * self.patch_W * self.C)
        self.num_patches = int(self.patch_dim * self.patch_dim)
        if self.verbose:
            print(" Patch Resolution (P): {}".format(self.patch_dim))
            print(" Number of Patches: {}".format(self.num_patches))
            print(" Length of Flattened Patch Vector: {}".format(self.patch_flat))
            print(" Patchification Output: (n, {}, {})".format(self.num_patches, self.patch_flat))

        # 2. Linear Projection (LP) of Flattened Patches
        self.LP_dim = config['LP']
        self.linear_projection = nn.Linear(self.patch_flat, self.LP_dim)
        if self.verbose:
            print(" Embedding Dimension: ", self.LP_dim)
            print(" Linear Projection Output: (n, {}, {})".format(self.num_patches, self.LP_dim))
        
        # 3. Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self.LP_dim))
        self.num_tokens = self.num_patches + 1
        if self.verbose:
            print(" Adding Class. Token Output: (n, {}, {})".format(self.num_tokens, self.LP_dim))

        # 4. Positional Embedding
        self.pos_embed = nn.Parameter(self.get_positional_embeddings(self.num_tokens, self.LP_dim))
        self.pos_embed.requires_grad = False
        if self.verbose:
            print(" Adding Positional Embed. Output: (n, {}, {})".format(self.num_tokens, self.LP_dim))

        # 5. Transformer Encoding Blocks
        self.num_head = config['num_head']
        self.num_enc_blocks = config['num_enc_blocks']
        self.mlp_ratio = config['mlp_ratio']
        self.ViTBlocks = nn.ModuleList([ViTBlock(self.LP_dim, self.num_head) 
                                        for _ in range(self.num_enc_blocks)])
        if self.verbose:
            print(" Number of Heads: ", self.num_head)
            print(" Number of Encoding Blocks: ", self.num_enc_blocks)
            print(" MLP Ratio: ", self.mlp_ratio)
            print(" ViT Encoding Output: (n, {}, {})".format(self.num_tokens, self.LP_dim))

        # 6. Classification Head
        self.out_dim = config['out_dim']
        self.cls_head = nn.Sequential(
            nn.Linear(self.LP_dim, self.out_dim),
            nn.Softmax(dim=-1)
        )
        if self.verbose:
            print(" Number of Classes: ", self.out_dim)
            print(" Classification Head Output: (n, {})".format(self.out_dim))

    def patchify(self, images):
        '''
        Patchify a batch of images into a sequence of patches.
        Output Dim. -> (N, PxP, HxC/P x WxC/P)
        '''
        n, c, h, w = images.shape
        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, self.num_patches, self.patch_flat)
        for idx, image in enumerate(images):
            for i in range(self.patch_dim):
                for j in range (self.patch_dim):
                    start_H, end_H = int(i * self.patch_H), int((i+1) * self.patch_H)
                    start_W, end_W = int(j * self.patch_W), int((j+1) * self.patch_W)
                    patch = image[:, start_H : end_H, start_W : end_W]
                    patches[idx, i * self.patch_dim + j] = patch.flatten()
        return patches
    
    def add_class_embedding(self, tokens):
        '''
        Add a classification token to the linear projected flattened patches

        Task specific token  add to our model that has the role of capturing information about the other tokens.
        When information about all other tokens will be present here, we will be able to classify the image using only this special token.
        
        Note: The classification token is put as the first token of each sequence. 
        '''
        return torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

    def get_positional_embeddings(self, num_tokens, LP_dim):
        '''
        Using Proposed Positional Encoding from Attention Is All You Need by Vaswani et al
        '''
        pos_embed = torch.ones(num_tokens, LP_dim)
        for i in range(num_tokens):
            for j in range(LP_dim):
                if j % 2 == 0:
                    pos_embed[i][j] = np.sin(i / (10000 ** (j / LP_dim)))
                else:
                    pos_embed[i][j] = np.cos(i / (10000 ** ((j - 1) / LP_dim)))
        return pos_embed.clone().detach()
    
    def add_positional_encoding(self, tokens, n):
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        return tokens + pos_embed

    def forward(self, images):
        patches = self.patchify(images)
        tokens = self.linear_projection(patches)
        tokens = self.add_class_embedding(tokens)
        tokens = self.add_positional_encoding(tokens, images.shape[0])
        
        for block in self.ViTBlocks:
            tokens = block(tokens)

        # Use classification token only (first token)
        cls_token = tokens[:, 0, :]
        out = self.cls_head(cls_token)
        return out

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_head):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head

        # Divide by number of head, such that the concatenated result
        # preserves the input embedded dimensionality
        assert embed_dim % num_head == 0, f"Can't divide dimension {embed_dim} into {num_head} heads"
        self.head_dim = int(embed_dim / num_head)
        
        self.q_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_head)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_head)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.num_head)])
        self.W_concat = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def self_attention(self, Q, k, v):
        attention = (self.softmax(Q @ k.transpose(1, 2) / (self.head_dim ** 0.5))) @ v
        return attention

    def forward(self, sequences):
        ''' naive implementation '''
        head_attentions = []
        batch_size, seq_len, emb_dim = sequences.size()
        sequences = sequences.view(batch_size, seq_len, self.num_head, self.head_dim)
        for head in range(self.num_head):
            W_q = self.q_mappings[head]
            W_k = self.k_mappings[head]
            W_v = self.v_mappings[head]
            head_seq = sequences[:, :, head, :]
            q, k, v = W_q(head_seq), W_k(head_seq), W_v(head_seq)
            head_attention = self.self_attention(q, k, v)
            head_attentions.append(head_attention)
        mha = torch.concat(head_attentions, -1)
        mha = self.W_concat(mha)
        return mha

class ViTBlock(nn.Module):
    def __init__(self, emb_dim, num_head, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = num_head

        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha_block = MultiheadAttention(emb_dim, num_head)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * emb_dim, emb_dim)
        )

    def forward(self, x):
        mha = self.mha_block(self.norm1(x))
        residual_mha = x + mha
        out = self.mlp(self.norm2(residual_mha))
        residual_out = residual_mha + out
        return residual_out