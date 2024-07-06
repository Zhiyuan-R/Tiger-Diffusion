import torch
import torch.nn as nn
import numpy as np
import math
import time
from timm.models.vision_transformer import PatchEmbed, Mlp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
def plot_att(att_tensor, T):

    att_arr = att_tensor.detach().cpu().numpy()[0][0]
    time_str = str(time.time())[-5:]
    filename = 'atten_map/ours/fig/' + time_str + '_' +str(T[0].item()) + '.png'
    arr_name = 'atten_map/ours/arr/' + time_str + '_' +str(T[0].item()) + '.npy'
    ax = sns.heatmap(att_arr)
    plt.savefig(filename)
    np.save(arr_name, att_arr)
    plt.clf()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.pe_proj_1 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GroupNorm(8,  dim),
            )
        
        self.pe_proj_2 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GroupNorm(8,  dim),
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pe=None, time=None, coor=None):
        B, N, C = x.shape
        pe_1 = self.pe_proj_1(pe.transpose(1, 2)).transpose(1, 2) # N, D
        pe_2 = self.pe_proj_2(pe.transpose(1, 2))  # D, N
        pe_attn = self.sigmoid(pe_1 @ pe_2)
        # qkv = (self.qkv(x) + pe.repeat(1, 1 ,3)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = (self.qkv(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # import pdb
        # pdb.set_trace()
        attn = attn * pe_attn[:,None,:,:]
        
        attn = attn.softmax(dim=-1)
        # plot_att(attn, time)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pe=None, time=None, coord = None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        att_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pe, time, coord)
        # att_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa.unsqueeze(1) * att_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(latent_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(latent_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_size, 2 * latent_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    
    def __init__(
        self,
        input_size=32,
        tok_num = 256,
        hidden_size=128,
        latent_size=512,
        output_channel=256,
        depth=36, #28
        num_heads=8, #8
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.tok_num = tok_num
        self.latent_size = latent_size

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_size, bias=True),
            nn.LayerNorm(latent_size)
        )
    
        self.t_embedder = TimestepEmbedder(latent_size)
        
        self.div_term = torch.exp(torch.arange(0, latent_size, 2) * (-math.log(10000.0) / latent_size))
        
        self.blocks = nn.ModuleList([
            DiTBlock(latent_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.FinalLayer = FinalLayer(latent_size, output_channel)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x, t, coords):
        """
        Forward pass of DiT.
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        t: (N, C, L) tensor of diffusion timesteps
        """
        time = t
        pe = torch.zeros(x.shape[0], self.tok_num, self.latent_size).to(x.device)
        div_term = self.div_term.to(x.device)
        coord_enc = coords[:, 0, :] * 10000 + coords[:, 1, :] * 100 + coords[:, 2, :]
        pe[:, :, 0::2] = torch.sin(coord_enc[:, :, None] * div_term)
        pe[:, :, 1::2] = torch.cos(coord_enc[:, :, None] * div_term)
        x = self.proj(torch.transpose(x, 2, 1).contiguous())  + pe
        t = self.t_embedder(t)                   # (N, D)
        c = t                                    # (N, D)
        for block in self.blocks:
            x = block(x, c, pe, time, coords)                      # (N, T, D)
            # x = block(x, c)        
        x = self.FinalLayer(x, c)
        x = torch.transpose(x, 2, 1).contiguous()# (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
