import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bit_plane_slicing, decode_bit_mask
from .luts import HDLUT, HDBLUT


class HKLUT(nn.Module): 
    def __init__(self, msb_weights, lsb_weights, msb='hdb', lsb='hd', upscale=2):
        super(HKLUT, self).__init__()
        self.upscale = upscale
        self.bit_mask = '11110000'
        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(self.bit_mask)

        # MSB
        msb_lut = HDLUT if msb=='hd' else HDBLUT
        self.msb_lut = msb_lut(*msb_weights, 2**self.msb_bits, upscale=upscale)


        # LSB
        lsb_lut = HDLUT if lsb=='hd' else HDBLUT
        self.lsb_lut = lsb_lut(*lsb_weights, 2**self.lsb_bits, upscale=upscale)


    def forward(self, img_lr):

        img_lr_255 = torch.floor(img_lr*255)
        img_lr_msb, img_lr_lsb = bit_plane_slicing(img_lr_255, self.bit_mask)

        # msb
        img_lr_msb = torch.floor_divide(img_lr_msb, self.msb_step)
        MSB_out = self.msb_lut(img_lr_msb)/255.


        # lsb
        img_lr_lsb = torch.floor_divide(img_lr_lsb, self.lsb_step)

        LSB_out = self.lsb_lut(img_lr_lsb)/255.

        img_out = MSB_out + LSB_out + nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr)
        
        return torch.clamp(img_out, 0, 1)

