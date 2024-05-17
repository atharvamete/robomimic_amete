import math
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

###############################################################################
#
# Positional Embedding module
#
###############################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:,:x.size(1), :]
        return self.dropout(pe.repeat(x.size(0),1,1))

###############################################################################
#
# MLP projection module
#
###############################################################################


class MLP_Proj(nn.Module):
    """
    Encode any embedding

    h = f(e), where
        e: embedding from some model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data)  # (B, H)
        return h

###############################################################################
#
# obs encoder modules
#
###############################################################################

class ObsEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoders = nn.ModuleList()
        for i in range(len(cfg.input_dim)):
            encoder = MLP_Proj(cfg.input_dim[i], cfg.output_dim[i], cfg.output_dim[i], num_layers=cfg.num_layers[i], dropout=cfg.dropout[i])
            self.encoders.append(encoder)
        self.encoders.append(MLP_Proj(sum(cfg.output_dim), cfg.proj_dim, cfg.proj_dim, num_layers=1))
        self.dropout = nn.Dropout(cfg.dropout[0])
    
    def forward(self, obs):
        x = [encoder(obs[i]) for i, encoder in enumerate(self.encoders[:-1])]
        x = torch.cat(x, dim=-1)
        x = self.dropout(x)
        x = self.encoders[-1](x)
        return x.unsqueeze(1)

###############################################################################
#
# 1D conv modules
#
###############################################################################

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, output_padding=stride-1)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)

class ResidualTemporalDeConvBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = DeConv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[::-1][i], 
                stride[::-1][i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size=sum(stride), stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = len(self.blocks)-1
        for block in self.blocks:
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num -= 1
            out = block(out)
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
    
###############################################################################
#
# Skill-VAE module
#
###############################################################################

class SkillVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.vq_type == 'vq':
            self.vq = VectorQuantize(dim=cfg.encoder_dim, codebook_dim=cfg.codebook_dim, codebook_size=cfg.codebook_size)
        elif cfg.vq_type == 'fsq':
            self.vq = FSQ(dim=cfg.encoder_dim, levels=cfg.fsq_level)
        else:
            raise NotImplementedError('Unknown vq_type')
        self.action_proj = nn.Linear(cfg.action_dim, cfg.encoder_dim)
        self.action_head = nn.Linear(cfg.decoder_dim, cfg.action_dim)
        self.conv_block = ResidualTemporalBlock(
            cfg.encoder_dim, cfg.encoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_encoder)
        self.deconv_block = ResidualTemporalDeConvBlock(
            cfg.decoder_dim, cfg.decoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_decoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.encoder_dim, nhead=cfg.encoder_heads, dim_feedforward=4*cfg.encoder_dim, dropout=cfg.attn_pdrop, activation='gelu', batch_first=True, norm_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.decoder_dim, nhead=cfg.decoder_heads, dim_feedforward=4*cfg.decoder_dim, dropout=cfg.attn_pdrop, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(cfg.encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(cfg.decoder_dim)
    
    def encode(self, act):
        x = self.action_proj(act)
        x = self.conv_block(x)
        x = self.add_positional_emb(x)
        if self.cfg.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.encoder(x, mask=mask, is_causal=True)
        else:
            x = self.encoder(x)
        return x

    def quantize(self, z):
        if self.cfg.vq_type == 'vq':
            codes, indices, commitment_loss = self.vq(z)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size).to(z.device)
        else:
            codes, indices = self.vq(z)
            commitment_loss = torch.tensor([0.0]).to(z.device)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size).to(z.device)
        pp_sample = torch.tensor(np.mean([len(torch.unique(index_seq)) for index_seq in indices])/z.shape[1]).to(z.device)
        return codes, indices, pp, pp_sample, commitment_loss

    def decode(self, codes):
        x = self.fixed_positional_emb(torch.zeros((codes.shape[0], self.cfg.skill_block_size, self.cfg.decoder_dim), dtype=codes.dtype, device=codes.device))
        if self.cfg.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.decoder(x, codes, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, codes)
        x = self.action_head(x)
        return x

    def forward(self, act):
        z = self.encode(act)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes)
        return x, pp, pp_sample, commitment_loss

    def get_indices(self, act):
        z = self.encode(act)
        _, indices, _, _, _ = self.quantize(z)
        return indices
    
    def decode_actions(self, indices):
        if self.cfg.vq_type == 'fsq':
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.get_output_from_indices(indices)
        x = self.decode(codes)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

###############################################################################
#
# Skill-GPT module
#
###############################################################################

class SkillGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size+1, cfg.n_embd)
        self.add_positional_emb = Summer(PositionalEncoding1D(cfg.n_embd))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.n_embd,
                nhead=cfg.n_head,
                dim_feedforward=4*cfg.n_embd,
                dropout=cfg.attn_pdrop,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=cfg.n_layer
        )
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.obs_encoder = ObsEncoder(cfg.encoder)
        if cfg.offset_layers > 0:
            self.offset_head = MLP_Proj(cfg.n_embd, cfg.offset_hidden_dim, cfg.offset_dim, num_layers=cfg.offset_layers)

    def forward(self, idx, context, targets=None, return_offset=False):
        x = self.tok_emb(idx)
        x = self.add_positional_emb(x)
        x = torch.cat([context, x], dim=1)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),x.device)
        x = self.decoder(x, mask=mask, is_causal=True)
        x = x[:, context.size(1):, :]
        x = self.lnf(x)
        logits = self.head(x)
        
        offset = self.offset_head(x[:,-1,:]) if return_offset else None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss
            return logits, loss, offset
        else:
            return logits, offset