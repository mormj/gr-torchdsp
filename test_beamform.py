
from torch import nn
import numpy as np
import torch


class Beamform(nn.Module):
    def __init__(self):
        super(Beamform, self).__init__()
        self.bf = torch.ones((1, 8), dtype=torch.float32)
        self.bf[-1,1::2] = 0.0

    def forward(
        self,
        in0,
        in1,
        in2,
        in3
    ):
        in_real = torch.stack((in0[::2], in1[::2], in2[::2],
                            in3[::2]), dim=0).reshape(-1, 4, 5)
        in_imag = torch.stack((in0[1::2], in1[1::2], in2[1::2],
                            in3[1::2]), dim=0).reshape(-1, 4, 5)

        bf_real = self.bf[0, ::2].reshape(-1, 1, 4)
        bf_imag = self.bf[0, 1::2].reshape(-1, 1, 4)
        out_real = torch.matmul(bf_real, in_real) - \
            torch.matmul(bf_imag, in_imag)
        out_imag = torch.matmul(bf_imag, in_real) + \
            torch.matmul(bf_real, in_imag)

        

        result = torch.cat([out_real, out_imag], dim=2)
        # result = torch.cat([in_real, in_imag], dim=0)
        result2 = torch.zeros_like(result)
        result2[:,::2] = result[:,:5]
        result2[:,1::2] = result[:,5:]
        # result = torch.stack((out_real,out_imag)).T.reshape(1,-1)
        # result = torch.cat([in_real, in_imag], dim=0)

        # in_matrix = torch.cat((in0, in1, in2, in3), dim=0).reshape(-1, 4, 1000)
        # out = torch.matmul(self.bf, in_matrix)
        # result = torch.cat([out.real, out.imag], dim=0)
        return result2


# 3 batches of iq interleaved (5 samples)
samp1 = torch.Tensor(range(10))
samp2 = torch.Tensor(range(10,20))
samp3 = torch.Tensor(range(20,30))

x = torch.stack((samp1,samp2,samp3)) #.unsqueeze(2)


model = Beamform()
model.eval()

model_output = model(x, x, x, x)

print(model_output)


# in0 = torch.Tensor([1,2,3,4,5,6])
# in1 = torch.Tensor([7,8,9,10,11,12])
# in2 = torch.Tensor([13,14,15,16,17,18])
# in3 = torch.Tensor([19,20,21,22,23,24])

# bf = torch.ones((1, 8), dtype=torch.float32)
# bf[-1,1::2] = 0.0

# in_real = torch.cat((in0[::2], in1[::2], in2[::2],
#                     in3[::2])).reshape(-1, 4, 3)
# in_imag = torch.cat((in0[1::2], in1[1::2], in2[1::2],
#                     in3[1::2])).reshape(-1, 4, 3)

# bf_real = bf[0, ::2].reshape(-1, 1, 4)
# bf_imag = bf[0, 1::2].reshape(-1, 1, 4)
# out_real = torch.matmul(bf_real, in_real) - \
#     torch.matmul(bf_imag, in_imag)
# out_imag = torch.matmul(bf_imag, in_real) + \
#     torch.matmul(bf_real, in_imag)

# result = torch.cat([out_real, out_imag], dim=0)