import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
import json
import os
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)

    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        emb = F.pad(emb, (0, 1))

    return emb


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, text_emb_dim, time_emb_dim, vocab_file=None):
        super().__init__()
        self.token2id = {}
        self.embedding = nn.Embedding(vocab_size, text_emb_dim)
        self.proj = nn.Linear(text_emb_dim, time_emb_dim)

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def build_vocab(self, texts, save_path=None):
        self.token2id = {'null': 0, ' ': 1}
        idx = 2
        for word in sorted(set(texts)):
            if word not in self.token2id:
                self.token2id[word] = idx
                idx += 1
        if save_path:
            self.save_vocab(save_path)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.token2id = json.load(f)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.token2id, f)

    def get_vocab_size(self):
        return len(self.token2id)

    def encode(self, text):
        if isinstance(text, str):
            words = text.split()
        else:
            words = text
        return [self.token2id.get(word, 0) for word in words]

    def forward(self, text_ids):
        if text_ids.dim() == 1:
            text_emb = self.embedding(text_ids)
        elif text_ids.dim() == 2:
            text_emb = self.embedding(text_ids).mean(dim=1)
        else:
            raise ValueError(f"text_ids must be 1D or 2D, got {text_ids.shape}")
        text_emb = self.proj(text_emb)
        return text_emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0, groups=32):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))

        scale_shift = self.time_mlp(t_emb)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(h)))

        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8, groups=32):
        super().__init__()

        if channels % groups != 0:
            for g in range(groups, 0, -1):
                if channels % g == 0:
                    groups = g
                    break
        self.norm = nn.GroupNorm(groups, channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.context_proj = nn.Linear(context_dim, channels)

    def forward(self, x, context):
        B, C, H, W = x.shape

        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)

        context = self.context_proj(context)

        attn_out, _ = self.attn(
            query=h,
            key=context,
            value=context
        )

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        return x + attn_out


class ResAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim, num_heads=8, dropout=0.0, groups=32):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim, dropout=dropout, groups=groups)
        self.attn = CrossAttention(out_channels, context_dim, num_heads=num_heads, groups=groups)

    def forward(self, x, t_emb, context=None):
        cond = t_emb
        if context is not None and context.dim() == 2:
            cond = cond + context
        x = self.res(x, cond)
        if context is not None and context.dim() == 3:
            x = self.attn(x, context)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels, time_emb_dim, context_dim, num_heads=8, dropout=0.0, groups=32):
        super().__init__()
        self.block1 = ResBlock(channels, channels, time_emb_dim, dropout=dropout, groups=groups)
        self.attn = CrossAttention(channels, context_dim, num_heads=num_heads, groups=groups)
        self.block2 = ResBlock(channels, channels, time_emb_dim, dropout=dropout, groups=groups)

    def forward(self, x, t_emb, context=None):
        cond = t_emb
        if context is not None and context.dim() == 2:
            cond = cond + context
        x = self.block1(x, cond)
        if context is not None and context.dim() == 3:
            x = self.attn(x, context)
        x = self.block2(x, cond)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=320,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        time_emb_dim=1280,
        text_emb_dim=768,
        groups=32,
        dropout=0.1,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.skip_channels = []
        channels = base_channels
        for mult in channel_mults:
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResAttentionBlock(channels, out_channels, time_emb_dim, text_emb_dim, dropout=dropout, groups=groups))
                self.skip_channels.append(out_channels)
                channels = out_channels
            self.down_blocks.append(Downsample(channels))

        self.bottleneck = Bottleneck(channels, time_emb_dim, text_emb_dim, dropout=dropout, groups=groups)

        self.up_blocks = nn.ModuleList()
        skip_channels = self.skip_channels.copy()
        for mult in reversed(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(ResAttentionBlock(channels + skip_ch, out_channels, time_emb_dim, text_emb_dim, dropout=dropout, groups=groups))
                channels = out_channels
            self.up_blocks.append(Upsample(channels))

        self.final_norm = nn.GroupNorm(groups, channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x, timesteps, context=None):
        t_emb = timestep_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        x = self.init_conv(x)
        skips = []

        for module in self.down_blocks:
            if isinstance(module, ResAttentionBlock):
                x = module(x, t_emb, context)
                skips.append(x)
            else:
                x = module(x)

        x = self.bottleneck(x, t_emb, context)

        for module in self.up_blocks:
            if isinstance(module, ResAttentionBlock):
                skip = skips.pop()
                if skip.shape[2:] != x.shape[2:]:
                    min_h = min(skip.shape[2], x.shape[2])
                    min_w = min(skip.shape[3], x.shape[3])
                    skip = skip[:, :, :min_h, :min_w]
                    x = x[:, :, :min_h, :min_w]
                x = torch.cat([x, skip], dim=1)
                x = module(x, t_emb, context)
            else:
                x = module(x)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        image_size=(16, 16),
        in_channels=3,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        base_channels=320,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        time_emb_dim=1280,
        groups=32,
        text_emb_dim=768,
        dropout=0.1,
        cfg_prob=0.1,
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            dropout=dropout,
            groups=groups
        )
        self.image_size = image_size
        self.channels = in_channels
        self.timesteps = timesteps
        self.cfg_prob = cfg_prob

        self.text_encoder = TextEncoder(vocab_size, text_emb_dim, time_emb_dim)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        return sqrt_ab * x0 + sqrt_1mab * noise

    def forward(self, x, text_ids):
        B = x.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x.device)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)

        if self.training:
            mask = torch.rand(B, device=x.device) < self.cfg_prob
            text_ids = text_ids.clone()
            text_ids[mask] = 0

        text_emb = self.text_encoder(text_ids)

        pred_noise = self.unet(x_t, t, text_emb)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, text_ids, guidance_scale=1.0):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

        text_emb = self.text_encoder(text_ids)

        if guidance_scale > 1.0:
            null_ids = torch.zeros_like(text_ids)
            uncond_emb = self.text_encoder(null_ids)
            eps_uncond = self.unet(x, t, uncond_emb)
            eps_cond = self.unet(x, t, text_emb)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = self.unet(x, t, text_emb)

        model_mean = (
            sqrt_recip_alpha_t *
            (x - beta_t / sqrt_one_minus_alpha_bar_t * eps)
        )

        if t[0] > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            return model_mean + sigma * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, batch_size, text_ids, guidance_scale=1.0):
        h, w = self.image_size
        x = torch.randn(batch_size, self.channels, h, w, device=device)

        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, text_ids, guidance_scale)

        return x


class ImageDataset(Dataset):
    def __init__(self, image_paths, text_tokens):
        self.image_paths = image_paths
        self.text_tokens = text_tokens

        print(f"Loaded {len(image_paths)} images and text tokens.")
        print(f"Example text token: {text_tokens[0]}")
        print(f"Example image path: {image_paths[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = img.resize((16, 16))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.
        img = img * 2 - 1
        text_token = self.text_tokens[idx]
        return img, torch.tensor(text_token, dtype=torch.long)

def show_images(x, nrow=None):
    if x.dim() == 3:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape
    x = x.detach().cpu()
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)

    if nrow is None:
        nrow = int(math.sqrt(B))
        nrow = max(1, nrow)

    grid = make_grid(x, nrow=nrow)
    grid = grid.permute(1, 2, 0).numpy()

    if C == 1:
        grid = grid.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def train_step(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for img, text_token in data_loader:
        img = img.to(device)
        text_token = text_token.to(device)

        optimizer.zero_grad()
        loss = model(img, text_token)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)
