from torch import nn
from scipy import signal as sp
import numpy as np
import torch


class MovingAverage(nn.Module):
    def __init__(self, taps: torch.tensor):
        super(MovingAverage, self).__init__()
        self.taps = taps

    def forward(self, iq_data):
        r = iq_data[:,:,::2]
        i = iq_data[:,:,1::2]
        
        result_r = nn.functional.conv1d(r, self.taps)
        result_i = nn.functional.conv1d(i, self.taps)

        x = torch.zeros(iq_data.shape[0], iq_data.shape[1], 1538, dtype=torch.float32)
        x[:,:,::2] = result_r
        x[:,:,1::2] = result_i

        return x


x = torch.randn(1, 1, 1024*2, requires_grad=False,
                dtype=torch.float32)

torch_taps = (1/100000.0) * torch.ones(1,256,
                dtype=torch.float32)
torch_taps.requires_grad = False
model = MovingAverage(torch_taps.reshape(1, 1, -1))

model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
