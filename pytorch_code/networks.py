"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import functools
from torchvision import models
##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
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
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
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
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
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
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

#        return relu5_3
        return [relu1_2, relu2_2, relu3_3, relu4_3, relu5_3]

# class Vgg19(nn.Module):
#     def __init__(self):
#         super(Vgg19, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

#     def forward(self, X):
#         h = F.relu(self.conv1_1(X), inplace=True)
#         h = F.relu(self.conv1_2(h), inplace=True)
#         relu1_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)

#         h = F.relu(self.conv2_1(h), inplace=True)
#         h = F.relu(self.conv2_2(h), inplace=True)
#         relu2_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)

#         h = F.relu(self.conv3_1(h), inplace=True)
#         h = F.relu(self.conv3_2(h), inplace=True)
#         relu3_2 = h
#         h = F.relu(self.conv3_3(h), inplace=True)
#         # relu3_3 = h
#         h = F.relu(self.conv3_4(h), inplace=True)
#         h = F.max_pool2d(h, kernel_size=2, stride=2)

#         h = F.relu(self.conv4_1(h), inplace=True)
#         h = F.relu(self.conv4_2(h), inplace=True)
#         relu4_2 = h
#         h = F.relu(self.conv4_3(h), inplace=True)
#         # relu4_3 = h
#         h = F.relu(self.conv4_4(h), inplace=True)

#         h = F.relu(self.conv5_1(h), inplace=True)
#         h = F.relu(self.conv5_2(h), inplace=True)
#         relu5_2 = h
#         h = F.relu(self.conv5_3(h), inplace=True)
#         # relu5_3 = h
#         h = F.relu(self.conv5_4(h), inplace=True)

# #        return relu5_3
#         return [relu1_2, relu2_2, relu3_2, relu4_2, relu5_2]


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_2 = h
        h = self.slice4(h)
        relu4_2 = h
        h = self.slice5(h)
        relu5_2 = h
        return [relu1_2, relu2_2, relu3_2, relu4_2, relu5_2]


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


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


class up2d(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size = 4,stride = 2,padding = 1):
        super(up2d, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.res_block1 = residual_block(out_ch)
        self.res_block2 = residual_block(out_ch)

    def forward(self, x1,x2,x3):
        up = self.upsample(x1)
        up = self.res_block1(up,x2)
        up = self.res_block2(up,x3)
        return up


class down2d(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size = 3,stride = 2,padding = 1):
        super(down2d, self).__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.res_block1 = residual_block(out_ch)
        self.res_block2 = residual_block(out_ch)

    def forward(self, x):
        x1 = self.downsample(x)
        x2 = self.res_block1(x1)
        x3 = self.res_block2(x2)
        return x1,x2,x3

class down2d_1res(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down2d_1res, self).__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.res_block1 = residual_block(out_ch)

    def forward(self, x):
        x1 = self.downsample(x)
        x2 = self.res_block1(x1)
        return x1,x2


class dec_block(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, use_dropout=False, pixel_shuffle=False):
        super(dec_block, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        self.model += [norm_layer(nc_output)]
        if use_dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_post_block(nn.Module):
    def __init__(self, nc_input, nc_output, pixel_shuffle=False, act_post=True):
        super(dec_post_block, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if act_post:
            self.model += [nn.Tanh()]
        else:
            self.model += [nn.Conv2d(nc_output, nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class enc_block_v3(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, inner_most=False):
        super(enc_block_v3, self).__init__()
        self.model = []
        self.model += [nn.Conv2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_block_v3(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, use_dropout=False, pixel_shuffle=False):
        super(dec_block_v3, self).__init__()
        self.model = []
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if use_dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_post_block_v3(nn.Module):
    def __init__(self, nc_input, nc_output, pixel_shuffle=False, act_post=True):
        super(dec_post_block_v3, self).__init__()
        self.model = []
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if act_post:
            self.model += [nn.Tanh()]
        else:
            self.model += [nn.Conv2d(nc_output, nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


# for batch_sizex3x128x128 input image
# the lowest resolution is 1
# the max num of filters is 512
# Unet as basic model

class Unet_V4_128x128(nn.Module):
    def __init__(self, nc_input_x, nc_input_y, nc_output):
        super(Unet_V4_128x128, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.struc_enc_pre_ = nn.Sequential(nn.Conv2d(nc_input_y, 64, kernel_size=4, stride=2, padding=1, bias=True))
        self.struc_enc_64_128_ = enc_block_v4(64, 128, norm_layer, inner_most=False)
        self.residual_0 = residual_block(128)
        self.struc_enc_128_256_ = enc_block_v4(128, 256, norm_layer, inner_most=False)
        self.residual_1 = residual_block(256)
        self.struc_enc_256_512_ = enc_block_v4(256, 512, norm_layer, inner_most=False)
        self.residual_2 = residual_block(512)
        self.struc_enc_512_512_0_ = enc_block_v4(512, 512, norm_layer, inner_most=False)
        self.residual_3 = residual_block(512)
        self.struc_enc_512_512_1_ = enc_block_v4(512, 512, norm_layer, inner_most=False)
        self.residual_4 = residual_block(512)
        self.struc_enc_inner_most_ = enc_block_v4(512, 512, norm_layer, inner_most=True)

        self.struc_dec_inner_most_ = dec_block_v4(512, 512, norm_layer, use_dropout=False, pixel_shuffle=True)
        self.residual_5 = residual_block(512)
        self.struc_dec_1024_512_1_ = dec_block_v4(1024, 512, norm_layer, use_dropout=True, pixel_shuffle=True)
        self.residual_6 = residual_block(512)
        self.struc_dec_1024_512_2_ = dec_block_v4(1024, 512, norm_layer, use_dropout=True, pixel_shuffle=True)
        self.residual_7 = residual_block(512)
        self.struc_dec_1024_256_ = dec_block_v4(1024, 256, norm_layer, use_dropout=False, pixel_shuffle=True)
        self.residual_8 = residual_block(256)
        self.struc_dec_512_128_ = dec_block_v4(512, 128, norm_layer, use_dropout=False, pixel_shuffle=True)
        self.residual_9 = residual_block(128)
        self.struc_dec_256_64_ = dec_block_v4(256, 64, norm_layer, use_dropout=False, pixel_shuffle=True)
        self.residual_10 = residual_block(64)
        self.struc_dec_post_ = dec_post_block_v4(128, 3, pixel_shuffle=True, act_post=False)
        self.struc_conv_1x1_level_0_block_0_ = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_0_block_1_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_1_block_0_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_1_block_1_ = nn.Sequential(nn.Conv2d(1536, 1024, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, y):

        ## structure-to-image unet pass
        # endoer
        y0 = self.struc_enc_pre_(y) # size: (n,64,64,64)
        y1 = self.struc_enc_64_128_(y0) # size: (n,128,32,32)
        y1 = self.residual_0(y1)
        y2 = self.struc_enc_128_256_(y1) # size: (n,256,16,16)
        y2 = self.residual_1(y2)
        y3 = self.struc_enc_256_512_(y2) # size: (n,512,8,8)
        y3 = self.residual_2(y3)
        y4 = self.struc_enc_512_512_0_(y3) # size: (n,512,4,4)
        y4 = self.residual_3(y4)
        y5 = self.struc_enc_512_512_1_(y4) # size: (n,512,2,2)
        y5 = self.residual_4(y5)
        y6 = self.struc_enc_inner_most_(y5) # size: (n,512,1,1)


        # # level 0 fusion
        # p0 = self.struc_conv_1x1_level_0_block_0_(y6) # size: (n,512,1,1)
        # # z_prior0 = self.latent_sample(p0) # size: (n,512,1,1)
        # h = torch.cat([y6, z_posterior0], 1) # size: (n,1024,1,1)
        # h = self.struc_conv_1x1_level_0_block_1_(h) # size: (n,512,1,1)

        # # level 1 fusion
        # h = self.struc_dec_inner_most_(h) # size: (n,512,2,2)
        # h = torch.cat([h, y5], 1) # size: (n,1024,2,2)
        # p1 = self.struc_conv_1x1_level_1_block_0_(h) # size: (n,512,2,2)
        # # z_prior1 = self.latent_sample(p1) # size: (n,512,2,2)
        # h = torch.cat([h, z_posterior1], 1) # size: (n,1536,2,2)
        # h = self.struc_conv_1x1_level_1_block_1_(h) # size: (n,1024,2,2)


        unet_dec_y = self.struc_dec_inner_most_(y6) # size: (n,512,2,2)
        unet_dec_y = self.residual_5(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y5], 1) # size: (n,1024,2,2)

        unet_dec_y = self.struc_dec_1024_512_1_(unet_dec_y)  # size: (n,512,4,4)
        unet_dec_y = self.residual_6(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y4], 1) # size: (n,1024,4,4)
        unet_dec_y = self.struc_dec_1024_512_2_(unet_dec_y)  # size: (n,512,8,8)
        unet_dec_y = self.residual_7(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y3], 1) # size: (n,1024,8,8)
        unet_dec_y = self.struc_dec_1024_256_(unet_dec_y)  # size: (n,256,16,16)
        unet_dec_y = self.residual_8(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y2], 1) # size: (n,512,16,16)
        unet_dec_y = self.struc_dec_512_128_(unet_dec_y)  # size: (n,128,32,32)
        unet_dec_y = self.residual_9(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y1], 1) # size: (n,256,32,32)
        unet_dec_y = self.struc_dec_256_64_(unet_dec_y)  # size: (n,64,64,64)
        unet_dec_y = self.residual_10(unet_dec_y)
        unet_dec_y = torch.cat([unet_dec_y, y0], 1) # size: (n,128,64,64)
        unet_dec_y = self.struc_dec_post_(unet_dec_y)  # size: (n,3,128,128)


        output = [unet_dec_y]
        # output = unet_dec_y

        return output


class enc_block_v4(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, inner_most=False):
        super(enc_block_v4, self).__init__()
        self.model = []
        self.model += [nn.Conv2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_block_v4(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, use_dropout=False, pixel_shuffle=False):
        super(dec_block_v4, self).__init__()
        self.model = []
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if use_dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_post_block_v4(nn.Module):
    def __init__(self, nc_input, nc_output, pixel_shuffle=False, act_post=True):
        super(dec_post_block_v4, self).__init__()
        self.model = []
        if pixel_shuffle:
            self.model += [nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if act_post:
            self.model += [nn.Tanh()]
        else:
            self.model += [nn.Conv2d(nc_output, nc_output, kernel_size=3, stride=1, padding=1, bias=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



class NetV2_128x128(nn.Module):
    def __init__(self, nc_input_x, nc_input_y, nc_output):
        super(NetV2_128x128, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.app_enc_pre_ = nn.Sequential(nn.Conv2d(nc_input_x, 64, kernel_size=3, stride=2, padding=1, bias=True))
        self.app_enc_64_128_ = enc_block_3x3(64, 128, norm_layer, inner_most=False)
        self.app_enc_128_256_ = enc_block_3x3(128, 256, norm_layer, inner_most=False)
        self.app_enc_256_512_ = enc_block_3x3(256, 512, norm_layer, inner_most=False)
        self.app_enc_512_512_0_ = enc_block_3x3(512, 512, norm_layer, inner_most=False)
        self.app_enc_512_512_1_ = enc_block_3x3(512, 512, norm_layer, inner_most=False)
        self.app_enc_inner_most_ = enc_block_3x3(512, 512, norm_layer, inner_most=True)

        self.app_dec_inner_most_ = dec_block_3x3(512, 512, norm_layer, use_dropout=False, inner_most=True)
        self.app_conv_1x1_level_0_block_0_ = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.app_conv_1x1_level_0_block_1_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.app_conv_1x1_level_1_block_0_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))


        self.struc_enc_pre_ = nn.Sequential(nn.Conv2d(nc_input_y, 64, kernel_size=3, stride=2, padding=1, bias=True))
        self.struc_enc_64_128_ = enc_block_3x3(64, 128, norm_layer, inner_most=False)
        self.struc_enc_128_256_ = enc_block_3x3(128, 256, norm_layer, inner_most=False)
        self.struc_enc_256_512_ = enc_block_3x3(256, 512, norm_layer, inner_most=False)
        self.struc_enc_512_512_0_ = enc_block_3x3(512, 512, norm_layer, inner_most=False)
        self.struc_enc_512_512_1_ = enc_block_3x3(512, 512, norm_layer, inner_most=False)
        self.struc_enc_inner_most_ = enc_block_3x3(512, 512, norm_layer, inner_most=True)

        self.struc_dec_inner_most_ = dec_block_3x3(512, 512, norm_layer, use_dropout=False)
        self.struc_dec_1024_512_1_ = dec_block_3x3(1024, 512, norm_layer, use_dropout=True)
        self.struc_dec_1024_512_2_ = dec_block_3x3(1024, 512, norm_layer, use_dropout=True)
        self.struc_dec_1024_256_ = dec_block_3x3(1024, 256, norm_layer, use_dropout=False)
        self.struc_dec_512_128_ = dec_block_3x3(512, 128, norm_layer, use_dropout=False)
        self.struc_dec_256_64_ = dec_block_3x3(256, 64, norm_layer, use_dropout=False)
        self.struc_dec_post_ = dec_post_block_3x3(128, 3)
        self.struc_conv_1x1_level_0_block_0_ = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_0_block_1_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_1_block_0_ = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True))
        self.struc_conv_1x1_level_1_block_1_ = nn.Sequential(nn.Conv2d(1536, 1024, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, y):

        ## appearance encoder pass
        # endoer
        x0 = self.app_enc_pre_(x) # size: (n,64,64,64)
        x1 = self.app_enc_64_128_(x0) # size: (n,128,32,32)
        x2 = self.app_enc_128_256_(x1) # size: (n,256,16,16)
        x3 = self.app_enc_256_512_(x2) # size: (n,512,8,8)
        x4 = self.app_enc_512_512_0_(x3) # size: (n,512,4,4)
        x5 = self.app_enc_512_512_1_(x4) # size: (n,512,2,2)
        x6 = self.app_enc_inner_most_(x5) # size: (n,512,1,1)

        # level 0 fusion
        q0 = self.app_conv_1x1_level_0_block_0_(x6) # size: (n,512,1,1)
        z_posterior0 = self.latent_sample(q0) # size: (n,512,1,1)
        h = torch.cat([x6, z_posterior0], 1) # size: (n,1024,1,1)
        h = self.app_conv_1x1_level_0_block_1_(h) # size: (n,512,1,1)

        # level 1 fusion
        h = self.app_dec_inner_most_(h) # size: (n,512,2,2)
        h = torch.cat([h, x5], 1) # size: (n,1024,2,2)
        q1 = self.app_conv_1x1_level_1_block_0_(h) # size: (n,512,2,2)
        z_posterior1 = self.latent_sample(q1) # size: (n,512,2,2)


        ## structure-to-image unet pass
        # endoer
        y0 = self.struc_enc_pre_(y) # size: (n,64,64,64)
        y1 = self.struc_enc_64_128_(y0) # size: (n,128,32,32)
        y2 = self.struc_enc_128_256_(y1) # size: (n,256,16,16)
        y3 = self.struc_enc_256_512_(y2) # size: (n,512,8,8)
        y4 = self.struc_enc_512_512_0_(y3) # size: (n,512,4,4)
        y5 = self.struc_enc_512_512_1_(y4) # size: (n,512,2,2)
        y6 = self.struc_enc_inner_most_(y5) # size: (n,512,1,1)


        # level 0 fusion
        p0 = self.struc_conv_1x1_level_0_block_0_(y6) # size: (n,512,1,1)
        # z_prior0 = self.latent_sample(p0) # size: (n,512,1,1)
        h = torch.cat([y6, z_posterior0], 1) # size: (n,1024,1,1)
        h = self.struc_conv_1x1_level_0_block_1_(h) # size: (n,512,1,1)

        # level 1 fusion
        h = self.struc_dec_inner_most_(h) # size: (n,512,2,2)
        h = torch.cat([h, y5], 1) # size: (n,1024,2,2)
        p1 = self.struc_conv_1x1_level_1_block_0_(h) # size: (n,512,2,2)
        # z_prior1 = self.latent_sample(p1) # size: (n,512,2,2)
        h = torch.cat([h, z_posterior1], 1) # size: (n,1536,2,2)
        h = self.struc_conv_1x1_level_1_block_1_(h) # size: (n,1024,2,2)


        unet_dec_y = self.struc_dec_1024_512_1_(h)  # size: (n,512,4,4)
        unet_dec_y = torch.cat([unet_dec_y, y4], 1) # size: (n,1024,4,4)
        unet_dec_y = self.struc_dec_1024_512_2_(unet_dec_y)  # size: (n,512,8,8)
        unet_dec_y = torch.cat([unet_dec_y, y3], 1) # size: (n,1024,8,8)
        unet_dec_y = self.struc_dec_1024_256_(unet_dec_y)  # size: (n,256,16,16)
        unet_dec_y = torch.cat([unet_dec_y, y2], 1) # size: (n,512,16,16)
        unet_dec_y = self.struc_dec_512_128_(unet_dec_y)  # size: (n,128,32,32)
        unet_dec_y = torch.cat([unet_dec_y, y1], 1) # size: (n,256,32,32)
        unet_dec_y = self.struc_dec_256_64_(unet_dec_y)  # size: (n,64,64,64)
        unet_dec_y = torch.cat([unet_dec_y, y0], 1) # size: (n,128,64,64)
        unet_dec_y = self.struc_dec_post_(unet_dec_y)  # size: (n,3,128,128)

        output = [q0, q1, p0, p1, unet_dec_y]
        # output = unet_dec_y

        return output

    def latent_sample(self, mean):
        stddev = 1.0
        eps = Variable(torch.randn(mean.size()).cuda(mean.data.get_device()))
        return mean + stddev * eps


class enc_block_3x3(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, inner_most=False):
        super(enc_block_3x3, self).__init__()
        self.model = []
        self.model += [nn.LeakyReLU(0.2, True)]
        self.model += [nn.Conv2d(nc_input, nc_output, kernel_size=3, stride=2, padding=1, bias=True)]
        if not inner_most:
            self.model += [norm_layer(nc_output)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_block_3x3(nn.Module):
    def __init__(self, nc_input, nc_output, norm_layer=nn.BatchNorm2d, use_dropout=False, inner_most=False):
        super(dec_block_3x3, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        if inner_most:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        self.model += [norm_layer(nc_output)]
        if use_dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class dec_post_block_3x3(nn.Module):
    def __init__(self, nc_input, nc_output):
        super(dec_post_block_3x3, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        self.model += [nn.Tanh()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
