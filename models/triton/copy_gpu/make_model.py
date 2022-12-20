from torch import nn
import numpy as np
import torch


class Copy(nn.Module):
    def __init__(self):
        super(Copy, self).__init__()

    def forward(
        self,
        x
    ):
        # in_real = torch.cat((in0[::2], ), dim=0).reshape(-1, 1, 1000)
        # in_imag = torch.cat((in0[1::2],), dim=0).reshape(-1, 1, 1000)

        #result = torch.cat([out_real, out_imag], dim=0)
        # result = torch.cat([in_real, in_imag], dim=0)
        # result = x + x

        # in_matrix = torch.cat((in0, in1, in2, in3), dim=0).reshape(-1, 4, 1000)
        # out = torch.matmul(self.bf, in_matrix)
        # result = torch.cat([out.real, out.imag], dim=0)
        return x


x = torch.randn(2000, requires_grad=False,
                dtype=torch.float32)

model = Copy()
model.eval()

model_output = model(x)
print(x.shape, model_output.shape, model_output.flatten()[:10])

scripted = torch.jit.trace(model, x)
scripted.save("1/model.pt")

scripted_output = scripted(x, )
print(scripted_output.shape)
