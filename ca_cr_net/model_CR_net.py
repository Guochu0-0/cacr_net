import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from ca_cr_net.networks import CA_CR
from ca_cr_net.loss import PerceptualLoss
from model_base import ModelBase
from metrics import PSNR, SSIM, SAM, MAE
from torch.optim import lr_scheduler

S1_BANDS = 2
S2_BANDS = 13
RGB_BANDS = 3

class ModelCRNet(ModelBase):
    def __init__(self, config):
        super(ModelCRNet, self).__init__()
        self.config = config
        self.net_G = CA_CR(config).cuda()
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=config.G_LR, betas=(config.BETA1, config.BETA2))

        self.net_G = nn.DataParallel(self.net_G)

        self.l1_loss = nn.L1Loss()
        if self.config.P_LOSS:
            self.pecpt_loss = PerceptualLoss()


    def set_input(self, input):
        self.cloudy_data = torch.stack(input['input']['S2'], dim=1).cuda()
        self.cloudfree_data = torch.stack(input['target']['S2'], dim=1).cuda()

        self.cloudy_name = os.path.splitext(os.path.basename(input['input']['S2 path'][0][0]))[0]
        in_S2_td    = input['input']['S2 TD']
        if self.config.BATCH_SIZE>1: in_S2_td = torch.stack((in_S2_td)).T

        if self.config.USE_SAR:
            in_S1_td    = input['input']['S1 TD']
            if self.config.BATCH_SIZE>1: in_S1_td = torch.stack((in_S1_td)).T
            self.sar_data = torch.stack(input['input']['S1'], dim=1).cuda()
            self.input_data = torch.cat([self.sar_data, self.cloudy_data], dim=2)
            self.dates = torch.stack((in_S1_td,in_S2_td)).float().mean(dim=0).cuda()
        else:
            self.input_data = self.cloudy_data
            self.dates = in_S2_td.float().cuda()

    def forward(self):
        pred_cloudfree_data = self.net_G(self.input_data)
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        self.optimizer_G.zero_grad()

        # g l1 loss ##     
        g_l1_loss = self.l1_loss(self.pred_cloudfree_data, self.cloudfree_data) * self.config.G2_L1_LOSS_WEIGHT
        loss_G = g_l1_loss

        # g content loss #
        if self.config.P_LOSS:
            rgb_ch = np.random.choice(S2_BANDS, RGB_BANDS, replace=False) # false rgb channel
            g_content_loss, g_mrf_loss = self.content_loss(self.pred_cloudfree_data.squeeze(1)[:, rgb_ch, ...], self.cloudfree_data.squeeze(1)[:, rgb_ch, ...])
            g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
            g_mrf_loss = g_mrf_loss * self.config.G2_STYLE_LOSS_WEIGHT

            p_loss = g_content_loss + g_mrf_loss
            loss_G = loss_G + p_loss

        loss_G.backward()
        self.optimizer_G.step()

        return loss_G.item()

    def val_scores(self):
        if self.config.vis:
            self.pred_cloudfree_data, vis = self.forward()
            self.val_imm_feat_save(vis)
        else:
            self.pred_cloudfree_data = self.forward()

        scores = {'PSNR': PSNR(self.pred_cloudfree_data, self.cloudfree_data),
                  'SSIM': SSIM(self.pred_cloudfree_data, self.cloudfree_data),
                  'SAM': SAM(self.pred_cloudfree_data, self.cloudfree_data),
                  'MAE': MAE(self.pred_cloudfree_data, self.cloudfree_data),
                  }
        return scores

    def val_img_save(self, epoch, idx):
        
        def enhance_contrast(image, alpha=1.5, beta=0.0):
            alpha = np.clip(alpha, 1.0, 3.0)
            beta = np.clip(beta, -1.0, 1.0)
            
            enhanced_image = np.clip(alpha * image + beta, 0, 1)
            
            return enhanced_image

        imgs_cloudy = []
        for i in range(self.config.INPUT_T):
            cloudy_i = self.cloudy_data[0, i, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
            imgs_cloudy.append(cloudy_i)

        gt = self.cloudfree_data[0, 0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        pred = self.pred_cloudfree_data[0, 0, [3, 2, 1], ...].permute(1, 2, 0).detach().cpu().numpy()
        sar = self.sar_data[0, 0, [0]].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()

#########################################################################################################################################
        # idx = 2197
        # alpha = 2
        # save_dir = os.path.join('vis', 'ablation', self.config.EXP_NAME, f'ImgNo_{idx}')
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # for i in range(self.config.INPUT_T):
        #     imgi = enhance_contrast(imgs_cloudy[i], alpha)
        #     plt.imsave(os.path.join(save_dir, f"cloudy_{i}.png"), imgi)

        # for i in range(self.config.INPUT_T):
        #     sar_i = self.sar_data[0, i, 0, ...].detach().cpu().numpy()
        #     plt.imsave(os.path.join(save_dir, f"sar_{i}.png"), sar_i)

        # pred = enhance_contrast(pred, alpha)
        # gt = enhance_contrast(gt, alpha)
        # plt.imsave(os.path.join(save_dir, f"pred.png"), pred) 
        # plt.imsave(os.path.join(save_dir, f"gt.png"), gt) 

#########################################################################################################################################


        merged1 = np.concatenate(imgs_cloudy, axis=1)

        merged1 = enhance_contrast(merged1, alpha=2)
        pred = enhance_contrast(pred, alpha=2)
        gt = enhance_contrast(gt, alpha=2)

        merged2 = np.concatenate([sar, pred, gt], axis=1)
        merged = np.concatenate([merged1, merged2], axis=1)

        save_dir = os.path.join('img_gen', self.config.EXP_NAME, f'epoch_{epoch}', f'{idx}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(os.path.join(save_dir, f"{self.cloudy_name}.png"), merged) 

        plt.imsave(os.path.join(save_dir, f"pred.png"), pred) 
        plt.imsave(os.path.join(save_dir, f"gt.png"), gt) 

        for i in range(self.config.INPUT_T):
            imgi = enhance_contrast(imgs_cloudy[i], alpha=2)
            plt.imsave(os.path.join(save_dir, f"cloudy_{i}.png"), imgi)

        for i in range(self.config.INPUT_T):
            sar_i = self.sar_data[0, i, 0, ...].detach().cpu().numpy()
            plt.imsave(os.path.join(save_dir, f"sar_{i}.png"), sar_i)

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G,  epoch, os.path.join(self.config.SAVE_MODEL_DIR, self.config.EXP_NAME))
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.config.SAVE_MODEL_DIR, self.config.CKPT_NAME, '%s_net_CR.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])

    # def scale_to_01(self, im, method):
    #     if method == 'default':
    #         # rescale from [-1, +1] to [0, 1]
    #         return (im + 1) / 2
    #     else:
    #         # dealing with only optical images, range 0,5
    #         # rescale from [0, 5] to [0, 1]
    #         return im / 5


    # def get_perceptual_loss(self):
    #     loss = 0.
    #     fake = self.netL(self.pred_cloudfree_data.squeeze(1))
    #     real = self.netL(self.cloudfree_data.squeeze(1))
    #     # pre-trained VGG16 expects input to be in [0, 1],
    #     # --> ResNet (baseline or initial model) input and target S2 patches are in [0, 5],
    #     #     STGAN (no pre-trained ResNet) input and target patches are in []
    #     #     outputs of STGAN (model resnet_9blocks) are in [-1, +1] via Tanh()
    #     #     outputs of 3D net (model ResnetGenerator3DWithoutBottleneck) are in [-1, +1] via Tanh()

    #     """
    #     if self.opt.alter_initial_model:
    #         # change 
    #         fake = self.netL(self.fake_B)
    #         real = self.netL(self.real_B, method)
    #     else:
    #         fake = self.netL(self.fake_B)
    #         real = self.netL(self.real_B, method)
    #     """
    #     mse = torch.nn.MSELoss()
    #     for i in range(len(fake)):
    #         loss += mse(fake[i], real[i])
    #     return loss
