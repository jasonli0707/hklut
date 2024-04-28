import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *


# HD
class HDUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3]}
    pad_dict = {'h': (0,1,0,0), 'd': (0,1,0,1)}
    avg_factor = 2.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()


        self.conv1 = Conv(1, nf, [1,2], stride=1, padding=0, dilation=1)
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)


    def dconv_forward(self, x, conv1):
        K = 2
        S = 1
        P = K - 1
        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)

        x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1]], dim=1) # d

        x = x.unsqueeze(1).unsqueeze(1)
        x = conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B, -1, (H - P) * (W - P))
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)

        if self.ktype=="d":
            x = self.dconv_forward(x_in, self.conv1)
        else: # "h"
            x = self.conv1(x_in)

        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)


        x = self.pixel_shuffle(x)

        if self.ktype == "h":
            x = x.reshape(B, C, self.upscale*(H), self.upscale*(W-1))
        elif self.ktype == "d":
            x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))
        else:
            raise AttributeError

        return torch.tanh(x)


    def get_lut_input(self, input_tensor):
        if self.ktype == "h":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 1, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        elif self.ktype == "d":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 2, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        else:
            raise AttributeError
        return input_tensor

# HDB
class HDBUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2)}
    avg_factor = 3.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()


        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h': # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'd': # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor