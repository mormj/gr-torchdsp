from torch import nn
import torch

import numpy as np

FFT_SIZE = 1024
CP_LEN = 256

# def diff(x, dim=-1, same_size=False):
#     # assert dim is -1, ‘diff only supports dim=-1 for now’
#     if same_size:
#         return F.pad(x[…,1:]-x[…,:-1], (1,0))
#     else:
#         return x[…,1:]-x[…,:-1]

# def unwrap(phi, dim=-1):
#     # assert dim is -1, ‘unwrap only supports dim=-1 for now’
#     dphi = diff(phi, same_size=False)
#     dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
#     dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
#     phi_adj = dphi_m-dphi
#     phi_adj[dphi.abs()<np.pi] = 0
#     return phi + phi_adj.cumsum(dim)




class OfdmSymbolRecovery(nn.Module):
    def __init__(self):
        super(OfdmSymbolRecovery, self).__init__()

        self.cp_len = CP_LEN
        self.t = torch.Tensor(list(range(FFT_SIZE)))
        self.offset = self.cp_len // 2

        self.pilot_spacing = 100
        
        self.nsyms = 824
        self.total_syms = self.nsyms + 1 + self.nsyms // self.pilot_spacing
        self.pad_l = (FFT_SIZE - self.total_syms) // 2
        self.pad_r = FFT_SIZE - self.pad_l - self.total_syms

        self.pilots_idx = list(range(self.pad_l,(FFT_SIZE-self.pad_r),self.pilot_spacing))

    def forward(self, in1):
        
        iq = torch.complex(in1[:,:,::2], in1[:,:,1::2])
        
        
        result = torch.roll( torch.fft.fft(
            iq[:,:,self.offset:self.offset+FFT_SIZE], dim=2, norm="backward"),
            FFT_SIZE // 2, dims=2
        )

        result = result * torch.exp(1j * self.offset * self.t * (2*np.pi/FFT_SIZE))

        a = torch.diff( torch.angle(result[:,:,self.pilots_idx]) , dim=2) 
        a = torch.where( a > np.pi, a-2*np.pi, a )
        a = torch.where( a < -np.pi, a+2*np.pi, a )

        mn = torch.mean(a, dim=2)

        result = result * torch.exp(-1j * self.t * mn / self.pilot_spacing)
        x = torch.zeros(result.shape[0], result.shape[1], self.nsyms*2, dtype=torch.float32)

        # remove cyclic prefix and pilots
        idx = 0
        jj = self.pad_l+1
        while idx < self.nsyms:
            n = min(self.pilot_spacing - 1, self.nsyms - idx)
            x[:,:,2*idx:2*(idx+n):2] = result[:,:,jj:jj+n].real
            x[:,:,1+2*idx:2*(idx+n):2] = result[:,:,jj:jj+n].imag
            idx += n
            jj += n+1

        return x



x1 = torch.randn(1, 1, (FFT_SIZE +CP_LEN)* 2, requires_grad=False,
                dtype=torch.float32)


model = OfdmSymbolRecovery()
model.eval()

print(x1.shape, model(x1).shape)

scripted = torch.jit.trace(model, [x1])
scripted.save("1/model.pt")
