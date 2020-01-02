"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, VAEGen, NetV2_128x128
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, load_vgg19, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class FACE_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(FACE_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        if hyperparameters['net_version'] == 'v2' and hyperparameters['crop_image_height'] == 128:
            self.gen = NetV2_128x128(hyperparameters['input_dim_a'], hyperparameters['input_dim_b'], hyperparameters['input_dim_a'])
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.l1loss = nn.L1Loss(size_average=True)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            if 'vgg_net' == 'vgg16':
                self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
                self.vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
            else:
                self.vgg = load_vgg19()
                self.vgg.eval()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input-target))

    def forward(self, x_a, x_b):
        self.eval()
        output = self.gen(x_a,x_b)
        self.train()
        return output

    def __latent_kl(self, p, q):
        mean_1 = p
        mean_2 = q
        kl_loss = 0.5 * torch.pow(mean_1-mean_2, 2)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def gen_update(self, x, y, hyperparameters):
        self.gen_opt.zero_grad()

        output = self.gen(x, y)
        self.loss_gen_recon = 0
        self.loss_gen_recon_kl_1 = 0
        self.loss_gen_recon_kl_2 = 0
        self.loss_gen_vgg = 0
        # reconstruction loss
        if hyperparameters['recon_w'] != 0:
            self.loss_gen_recon = self.recon_criterion(output[-1], x)
        if hyperparameters['kl_w'] != 0:
            self.loss_gen_recon_kl_1 = self.__latent_kl(output[0], output[2])
            self.loss_gen_recon_kl_2 = self.__latent_kl(output[1], output[3])
        # perceptual loss
        if hyperparameters['vgg_w'] != 0:
            self.loss_gen_vgg = self.compute_vgg_loss(self.vgg, output[-1], x, hyperparameters) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        # self.loss_gen_total = hyperparameters['recon_w'] * self.loss_gen_recon + \
        #                       hyperparameters['kl_w'] * self.loss_gen_recon_kl_1 + \
        #                       hyperparameters['kl_w'] * self.loss_gen_recon_kl_2 + \
        #                       hyperparameters['vgg_w'] * self.loss_gen_vgg
        self.loss_gen_total = hyperparameters['recon_w'] * self.loss_gen_recon + \
                              hyperparameters['kl_w'] * self.loss_gen_recon_kl_1 + \
                              hyperparameters['kl_w'] * self.loss_gen_recon_kl_2 + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target, hyperparameters):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        total_loss = 0
        for i in range(len(img_fea)):
            total_loss += hyperparameters['feature_weights'][i] * torch.mean(torch.abs(img_fea[i]-target_fea[i]))
        return total_loss

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def transfer(self,x_a,x_b):
        self.eval()
        out = self.gen(x_a,x_b)
        self.train()
        return out

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen_weight'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.gen_opt.load_state_dict(state_dict['gen'])

        # Reinitilize schedulers
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen_weight': self.gen.state_dict()}, gen_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
