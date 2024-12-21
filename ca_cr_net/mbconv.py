import torch
import torch.nn as nn

S2_BANDS = 13


def get_norm_layer(out_channels, num_feats, n_groups=4, layer_type='batch'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)

class ResidualConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        n_groups=4,
        #last_relu=True,
        k=3, s=1, p=1,
        padding_mode="reflect",
    ):
        super(ResidualConvBlock, self).__init__(pad_value=pad_value)

        self.conv1 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv3 = ConvLayer(
            nkernels=nkernels,
            #norm='none',
            #last_relu=False,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )

    def forward(self, input):

        out1 = self.conv1(input)        # followed by built-in ReLU & norm
        out2 = self.conv2(out1)         # followed by built-in ReLU & norm
        out3 = input + self.conv3(out2) # omit norm & ReLU
        return out3

class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3, s=1, p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu: # append a ReLU after the current CONV layer
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2: # only append ReLU if not last layer
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, n_groups=4):
        super().__init__()
        self.norm = get_norm_layer(dim, dim, n_groups, norm)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(TemporallySharedBlock):
    def __init__(self, inp, oup, downsample=False, expansion=4, norm='batch', n_groups=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                          padding=1, padding_mode='reflect', groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride=stride, padding=0, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, padding_mode='reflect',
                          groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm), 
            )
        
        self.conv = PreNorm(inp, self.conv, norm, n_groups=4)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class Compact_Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Compact_Temporal_Aggregator, self).__init__()
        self.mode = mode
        # moved dropout from ScaledDotProductAttention to here, applied after upsampling 
        self.attn_dropout = nn.Dropout(0.1) # no dropout via: nn.Dropout(0.0)

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out