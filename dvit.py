import torch, os
import numpy as np
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

import torch.nn.functional as F

from itertools import repeat

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import collections.abc
from functools import partial

from torch.jit import Final

torch.manual_seed(42)
torch.cuda.manual_seed(42) 


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        try:
            return tuple(repeat(x, n))
        except:
            print('X is not the right type ', type(x))
    return parse


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,         
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
_EXPORTABLE = False
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def modulate(x, shift, scale, fin=False):
    if fin:
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, fin=True)
        x = self.linear(x)
        return x

import math
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class AgeEmbedder(nn.Module):
    """
    Embeds scalar ages into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def age_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        a_freq = self.age_embedding(t, self.frequency_embedding_size)
        return self.mlp(a_freq)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
class PatchEmbedder(nn.Module):
    def __init__(self, C, img_size, embed_dim, patch_size):
        super().__init__()
        self.img_size = img_size
        self.C = C
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(self.C, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        B, C, F, H, W, L = x.size()
        assert H==W==L== self.img_size
        assert C==self.C
        patches=[]
        for _ in range(F):
            patches.append(self.proj(x[:,:,_,...]))
        return torch.stack(patches, dim=1).view(B,F,self.embed_dim,-1).permute(0,1,3,2)
    
class DePatchEmbedder(nn.Module):
    def __init__(self, C, img_size, embed_dim, patch_size):
        super().__init__()
        self.img_size = img_size
        self.C = C
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.un_proj = nn.ConvTranspose3d(self.embed_dim, self.C, kernel_size=self.patch_size, stride=self.patch_size, bias=True)

    def forward(self, x):
        B, F, N, D = x.size()
        assert D==self.embed_dim
        patches=[]
        for _ in range(F):
            y = x[:,_,...].permute(0,2,1)
            y = y.view(B, D, self.img_size//self.patch_size, self.img_size//self.patch_size, self.img_size//self.patch_size)
            patches.append(self.un_proj(y))
        return torch.stack(patches, dim=2)
    
class CondPatchEmbedder(nn.Module):
    def __init__(self, C, img_size, embed_dim, patch_size):
        super().__init__()
        self.img_size = img_size
        self.C = C
        self.num_patches = (img_size//patch_size)**3
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(self.C, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.c_linear = nn.Linear(self.num_patches*self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, C, H, W, L = x.size()
        assert H==W==L== self.img_size
        assert C==self.C
        patches = self.c_linear(torch.flatten(self.proj(x), start_dim=1))
        return patches
    
class DViT(nn.Module):
    def __init__(self, 
                  patch_size=4, 
                  embed_dim=None, 
                  num_patches =None,
                  img_size=32,
                  channels=3,
                  num_heads=4, 
                  mlp_ratio=4.0,
                  num_classes=3,
                  depth=3,
                  num_frames = 3,
                  out_channels = 3):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.training = False
        self.num_frames = num_frames
        self.out_channels = out_channels
        self.depth = depth

        if not self.embed_dim:
            self.embed_dim = (self.patch_size**3) * self.channels
        if not self.num_patches:
            self.num_patches = (self.img_size//self.patch_size)**3

        self.y_embedder = LabelEmbedder(self.num_classes, self.embed_dim, dropout_prob=0)
        self.t_embedder = TimestepEmbedder(self.embed_dim, self.embed_dim)
        self.a_embedder = AgeEmbedder(self.embed_dim, self.embed_dim)
        self.patch_embedder = PatchEmbedder(self.channels, self.img_size, self.embed_dim, self.patch_size)
        self.cond_patch_embedder = CondPatchEmbedder(self.channels, self.img_size, self.embed_dim, self.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, self.num_frames, self.embed_dim), requires_grad=False)

        self.spatial_blocks = nn.ModuleList([DiTBlock(self.embed_dim, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])
        
        self.temporal_blocks = nn.ModuleList([DiTBlock(self.embed_dim, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])
        
        self.final_layer = FinalLayer(self.embed_dim, patch_size, self.out_channels)
        self.cond_tan = nn.Tanh()
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = self.get_2d_sincos_pos_embed(self.img_size//self.patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = self.get_1d_sincos_temp_embed(self.embed_dim, self.num_frames)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
# 
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embedder.proj.bias, 0)

        w = self.cond_patch_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_patch_embedder.proj.bias, 0)

        w = self.cond_patch_embedder.c_linear.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_patch_embedder.c_linear.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.a_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.a_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.spatial_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.temporal_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega 

        pos = pos.reshape(-1)  
        out = np.einsum('m,d->md', pos, omega) 

        emb_sin = np.sin(out) 
        emb_cos = np.cos(out) 

        emb = np.concatenate([emb_sin, emb_cos], axis=1) 
        return emb

    def get_1d_sincos_temp_embed(self, embed_dim, length):
        pos = torch.arange(0, length).unsqueeze(1)
        return self.get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


    def get_2d_sincos_pos_embed_from_grid(self, grid):
        assert self.embed_dim % 3 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(self.embed_dim // 3, grid[0])  # (H*W*L, D/3)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(self.embed_dim // 3, grid[1])  # (H*W*L, D/3)
        emb_l = self.get_1d_sincos_pos_embed_from_grid(self.embed_dim // 3, grid[2])  # (H*W*L, D/3)

        emb = np.concatenate([emb_h, emb_w, emb_l], axis=1) # (H*W*L, D)
        return emb


    def get_2d_sincos_pos_embed(self, grid_size, cls_token=False, extra_tokens=0):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid_l = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h, grid_l)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(grid)
        return pos_embed
    


    def unpatchify(self, x):
        B, F, N, D = x.size(); C=self.out_channels
        p_dim = self.img_size//self.patch_size
        x = x.reshape(shape=(B, F, p_dim, p_dim, p_dim, self.patch_size, self.patch_size, self.patch_size, C))
        x = torch.einsum('nfhwlpqrc->ncfhpwqlr', x)
        imgs = x.reshape(shape=(x.shape[0], C, F, p_dim * self.patch_size, p_dim * self.patch_size, p_dim * self.patch_size))
        return imgs


    def forward(self, x, t, y, img_cond, age_cond):
        self.training = True
        B,C,F,H,W,L = x.size()
        assert H==W==L== self.img_size
        x = self.patch_embedder(x)   
        a_embd=[]
        for _ in range(age_cond.size(1)):
            a_embd.append(self.a_embedder(age_cond[:,_]))   
        a_embd_inp = torch.stack(a_embd[1:], dim=1).unsqueeze(2)
        x = (x + a_embd_inp)/2
        x = x + self.pos_embed
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        i = self.cond_patch_embedder(img_cond)
        c = (t + y + i + a_embd[0])/4
        

        for d_, spat_transformer, temp_transformer in zip(range(self.depth), self.spatial_blocks, self.temporal_blocks):
            B,F,N,D = x.size() 
            x = rearrange(x, 'b f n d -> (b f) n d')
 
            # attend across space
            x = spat_transformer(x, repeat(c, 'b n -> (b repeat) n', repeat=F))
            x = rearrange(x, '(b f) n d -> b f n d', b = B)
            
            # attend across time
            x = rearrange(x, 'b f n d -> (b n) f d') 
            x = x + self.temp_embed
            x = temp_transformer(x, repeat(c, 'b n -> (b repeat) n', repeat=N))
            x = rearrange(x, '(b n) f d -> b f n d', b = B)

        x = self.final_layer(x, c)                
        x = self.unpatchify(x)
        return x
