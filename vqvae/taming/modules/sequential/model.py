# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        if channels == 1:
            num_groups = 1
        else:
            num_groups=4
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, dropout=0):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.Dropout(dropout),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


class SeqClassification(nn.Module):
    def __init__(self, z_dim, num_classes=3):
        super().__init__()
        self.z_dim = z_dim
        self.lin_c = nn.Linear(z_dim, num_classes)
    
    def forward(self, x):
        x = x[:,:, 0, 0]
        return self.lin_c(x)
    

class SeqEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, time_steps=None, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_veh = 9
        self.nonlinearity = Swish()
        self.time_steps = time_steps
        
        # B x N x T x F
        self.in_resblock = ResnetBlock(in_channels, in_channels)
        self.attn0 = NonLocalBlock(in_channels)
        
        ch = int(in_channels*2)
        self.mid_resblock = ResnetBlock(in_channels, ch)
        self.attn1 = NonLocalBlock(ch)
        
        self.out_resblock = ResnetBlock(ch, in_channels)
        self.out_lin1 = nn.Linear(in_channels*time_steps*self.num_veh, 
                                  int(0.5*in_channels)*time_steps*self.num_veh)
        self.attn2 = NonLocalBlock(1)
        self.out_lin2 = nn.Linear(int(0.5*in_channels)*time_steps*self.num_veh, 
                                  z_channels)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        # timestep embedding
        temb = None
        b, c, n, t = x.shape
        
        # input
        h = self.in_resblock(x)
        h = self.attn0(h)
        h = self.nonlinearity(h)
        
        # middle
        h = self.mid_resblock(h)
        h = self.attn1(h)
        
        # out
        h = self.out_resblock(h)
        hs = h.reshape(b, -1)
        h = self.out_lin1(hs)
        # h = self.attn2(h)
        h = self.nonlinearity(h)
        h = self.out_lin2(h)
        return h[:,:, None, None]

class SeqDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, time_steps=None, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.nonlinearity = Swish()
        self.time_steps = time_steps
        self.classificator = SeqClassification(z_channels)
        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        intermed = int(2*z_channels)
        self.conv_in = nn.Linear(z_channels, intermed)

        # middle
        block_in = 4
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = NonLocalBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        out_channels = block_in
        self.up = nn.Sequential(
            nn.Linear(intermed, int(2*intermed)),
            Swish())
        self.up_conv = nn.Sequential(
            nn.Conv2d(block_in, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.Dropout(dropout),
            Swish(),
        )


        # end
        self.lin_up = nn.Linear(512, 1024)


        'modified to fit dimensions'
        self.conv_out = nn.Linear(1024, out_ch*9*time_steps)

    def forward(self, z):
        c = self.classificator(z)
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        b, _ = z.shape[0], z.shape[1]

        # z to block_in
        h = self.conv_in(z[:,:,0,0]).reshape(b, 4, -1, 1)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h).reshape(b,-1)

        # upsampling
        h = self.up(h)
        
        
        # end
        if self.give_pre_end:
            return h

        h = self.lin_up(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h).reshape(b, self.in_channels, 9, self.time_steps)
        return h, c






