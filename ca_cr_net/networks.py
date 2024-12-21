import torch
import torch.nn as nn
# import sys
# sys.path.append('/remote-home/chuguoyou/Code/CR/CR')
import math
from einops.layers.torch import Rearrange
from ca_cr_net import transformer
from ca_cr_net.ltae import LTAE2dtiny
from ca_cr_net.utils import PatchPositionEmbeddingSine
from ca_cr_net.mbconv import Compact_Temporal_Aggregator, MBConv

S1_BANDS = 2
S2_BANDS = 13

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class CA_CR(nn.Module):
    def __init__(self, config):
        super(CA_CR, self).__init__()
        dim = 256
        self.config = config

        self.input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4)
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
            nn.Linear(4*4*13, dim)
        )
        self.opt_enc = transformer.TransformerEncoders(dim, nhead=2, num_encoder_layers=9, dim_feedforward=dim*2, activation='gelu', 
                                                               withCDP=self.config.with_CDP)
        self.cnn_dec = CNNDecoder(256, 13, 'ln', 'lrelu', 'reflect')

        self.sar_enc = CNNEncoder(2, 256, 'group', 'lrelu', 'reflect')
        self.init_sar_enc()

        self.late_conv_fuse_layer = nn.Sequential(nn.Conv2d(512, 256, 1, 1),
                                                  MBConv(256, 256, expansion=2, norm='group'))

        if self.config.useCA:
            self.t_fuse_blk = SG_TAggregator()
        else:
            self.t_fuse_blk = TAggregator()

    def forward(self, inputs):

        b, t, _, h, w = inputs.shape
        inputs_sar = inputs[:,:,:S1_BANDS,...].view(-1, S1_BANDS, h, w)
        inputs_opt = inputs[:,:,S1_BANDS:,...].view(-1, S2_BANDS, h, w)

        input_pos = self.input_pos.unsqueeze(0).repeat(inputs_opt.shape[0], 1, 1, 1).cuda()
        input_pos = input_pos.flatten(2).permute(2, 0, 1)
        patch_embedding = self.patch_to_embedding(inputs_opt)

        opt_feat = self.opt_enc(patch_embedding.permute(1, 0, 2), src_pos=input_pos)
        sar_feat = self.sar_enc(inputs_sar)

        bs, L, C  = patch_embedding.size()
        opt_feat = opt_feat.permute(1,2,0).view(b*t, C, int(math.sqrt(L)), int(math.sqrt(L)))
        enhanced_opt_feat = self.late_conv_fuse_layer(torch.cat([opt_feat, sar_feat], dim=1))

        enhanced_opt_feat = enhanced_opt_feat.view(b, t, C, int(math.sqrt(L)), int(math.sqrt(L)))
        
        if self.config.useCA:
            sar_feat = sar_feat.view(b, t, C, int(math.sqrt(L)), int(math.sqrt(L)))
            feat = self.t_fuse_blk(enhanced_opt_feat, sar_feat)
        else:
            feat = self.t_fuse_blk(enhanced_opt_feat)            

        output = self.cnn_dec(feat)

        return output.unsqueeze(1)

    def init_sar_enc(self):
        sar_enc_w = torch.load(self.config.PRETRAINED_VQVAE_PATH)

        encoder_dict = {k: v for k, v in sar_enc_w.items() if k.startswith('_encoder.')}
        new_encoder_dict = {}
        for k, v in encoder_dict.items():
            if k.startswith('_encoder.'):
                new_key = k[len('_encoder.'):]
                new_encoder_dict[new_key] = v
            else:
                new_encoder_dict[k] = v

        self.sar_enc.load_state_dict(new_encoder_dict)

        for param in self.sar_enc.parameters():
            param.requires_grad = False

########################################################################################################################

class SG_TAggregator(nn.Module):
    def __init__(self):
        super(SG_TAggregator, self).__init__()

        self.temporal_encoder = LTAE2dtiny(
            in_channels=256,
            d_model=256,
            n_head=16,
            d_k=4,
            positional_encoding=False,
        )
        self.temporal_aggregator = Compact_Temporal_Aggregator(mode='att_group')

        self.conv_fuse_layer = nn.Sequential(nn.Conv2d(256+1, 256, 1, 1),
                                             MBConv(256, 256, expansion=2, norm='group'))

    def forward(self, opt_feat, sar_feat):

        b, t, c, h, w = opt_feat.shape

        sar_temp_sim = self.temporal_cosine_similarity(sar_feat).view(b*t, 1, h, w)
        opt_feat_ = opt_feat.view(b*t, c, h, w)

        feat_fused = self.conv_fuse_layer(torch.cat([sar_temp_sim, opt_feat_], dim=1))

        att_down = 32
        down = nn.AdaptiveMaxPool2d((att_down, att_down))(feat_fused).view(b, t, c, att_down, att_down)
        att = self.temporal_encoder(down)
        opt_feat = self.temporal_aggregator(opt_feat, attn_mask=att)

        return opt_feat

    def temporal_cosine_similarity(self, feat):

        feat_norm = feat / (torch.norm(feat, dim=2, keepdim=True) + 1e-8)

        first_frame = feat_norm[:, 0:1, :, :, :]
        
        similarity = torch.sum(feat_norm * first_frame, dim=2)

        return similarity.unsqueeze(2)

# navie L-TAE for ca ablation
class TAggregator(nn.Module):
    def __init__(self):
        super(TAggregator, self).__init__()

        self.temporal_encoder = LTAE2dtiny(
            in_channels=256,
            d_model=256,
            n_head=16,
            d_k=4,
            positional_encoding=False,
        )
        self.temporal_aggregator = Compact_Temporal_Aggregator(mode='att_group')

    def forward(self, opt_feat):

        b, t, c, h, w = opt_feat.shape
        opt_feat_ = opt_feat.view(b*t, c, h, w)
        feat_fused = opt_feat_

        att_down = 32
        down = nn.AdaptiveMaxPool2d((att_down, att_down))(feat_fused).view(b, t, c, att_down, att_down)
        att = self.temporal_encoder(down)
        opt_feat = self.temporal_aggregator(opt_feat, attn_mask=att)

        return opt_feat

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activ, pad_type):
        super(CNNEncoder, self).__init__()

        dim = output_dim // 4
        self.conv1 = Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            Conv2dBlock(dim, dim * 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            Conv2dBlock(dim *2, dim * 4, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.conv4 = nn.Sequential(
            MBConv(256, 256, downsample=False, expansion=2, norm=norm),
            MBConv(256, 256, downsample=False, expansion=2, norm=norm)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        output = self.conv4(x3)
        return output

class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activ, pad_type):
        super(CNNDecoder, self).__init__()
        self.model = []
        dim = input_dim

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        dim //= 2
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.conv3 = Conv2dBlock(dim//2, output_dim, 5, 1, 2, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = self.conv3(x2)
        return output

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'instance':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'group':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import argparse
    import os
    from config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='myNet/myNet_sar_vqvae', help='the name of config files')
    opts = parser.parse_args()
    config = Config(os.path.join('config', f'{opts.config_name}.yml'))

    model = CA_CR(config)#.cuda()
    print(model)

    input = torch.rand(2, 3, 15, 256, 256)#.cuda()
    #dates = torch.rand(4, 3)

    output1, output2 = model(input)

    print(output2.shape)