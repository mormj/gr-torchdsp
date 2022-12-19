from torch import nn
import torch

OP_SIZE = 512


class MultiplyConj(nn.Module):
    def __init__(self):
        super(MultiplyConj, self).__init__()

    def forward(self, iq_data0, iq_data1):
        iq0 = torch.complex(iq_data0[:,:,::2], iq_data0[:,:,1::2])
        iq1 = torch.complex(iq_data1[:,:,::2], iq_data1[:,:,1::2])
        result = torch.multiply(iq0, torch.conj(iq1))
        r = result.real
        i = result.imag

        x = torch.zeros(result.shape[0], result.shape[1], result.shape[2]*2, dtype=torch.float32)
        x[:,:,::2] = r
        x[:,:,1::2] = i

        return x



x1 = torch.randn(1, 1, OP_SIZE*2, requires_grad=False,
                dtype=torch.float)
x2 = torch.randn(1, 1, OP_SIZE*2, requires_grad=False,
                dtype=torch.float)

model = MultiplyConj()
model.eval()

print(x1.shape, model(x1,x2).shape)

scripted = torch.jit.trace(model, [x1,x2])
scripted.save("1/model.pt")