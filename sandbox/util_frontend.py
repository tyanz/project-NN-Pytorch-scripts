"""
util_frontend.py

Utilities for frontend feature extraction

LFCC: asvspoof.org baseline matlab code

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import sandbox.util_dsp as nii_dsp
import core_scripts.data_io.conf as nii_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##################
## other utilities
##################
def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
    
    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)
    y = torch.zeros_like(x, dtype=nii_conf.d_dtype)
    if a < b:
        index = torch.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:    
        index = torch.logical_and(b < x, x < c)              
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y 
    
def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)
    
    output
    ------
    output (batch, Length, dim)
    
    Delta is calculated along Length
    """
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = torch_nn_func.pad(x.unsqueeze(1), (0, 0, 1, 1), 
                               'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:,2:]
    return output

#################
## LFCC front-end
#################

class LFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code
    Difference: 1. with_energy is added to set the first dimension as energy
    
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True):
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        
        f = (sr / 2) * torch.linspace(0, 1, fn//2+1)
        filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
        filter_bank = torch.zeros([fn//2+1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx], 
                    filter_bands[idx+1], 
                    filter_bands[idx+2]])
        self.lfcc_fb = torch_nn.Parameter(filter_bank, requires_grad=False)
        self.l_dct = nii_dsp.LinearDCT(filter_num, 'dct', norm='ortho')
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        return
    
    def forward(self, x):
        """
        x: tensor(batch, length)
        """
        # pre-emphsis 
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        
        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl, 
                            window=torch.hamming_window(self.fl), 
                            onesided=True, pad_mode="constant")        
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        # CEP
        cep = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) + 
                          torch.finfo(torch.float32).eps)
        
        # DCT
        lfcc = self.l_dct(cep)
        
        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy
            
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc
        return lfcc_output