from torch import nn
import torch

FFT_SIZE = 64


class OfdmDivide(nn.Module):
    def __init__(self):
        super(OfdmDivide, self).__init__()

    def forward(self, in1, in2):
        
        iq1 = torch.complex(in1[:,:,::2], in1[:,:,1::2])
        iq2 = torch.complex(in2[:,:,::2], in2[:,:,1::2])
        
        iq3 = torch.ones_like(iq1)

        # should be able to do this, but not supported by torchscript
        # result = torch.div(in1.view(torch.cfloat), iq2.view(torch.cfloat))
        result = torch.div(iq1, iq2)
        # We do this because TIS doesn't like complex outputs sometimes
        # result = torch.cat([result.real, result.imag], dim=1)
        # return result.permute((0, 2, 1))
        r = result.real
        i = result.imag
        # r = iq1.real
        # i = iq1.imag

        # Zero pad the output to go from e.g. 64 --> 128
        x = torch.zeros(result.shape[0], result.shape[1], result.shape[2]*2*2, dtype=torch.float32)
        # x = torch.zeros(iq1.shape[0], iq1.shape[1], iq1.shape[2]*2*2, dtype=torch.float32)
        x[:,:,:FFT_SIZE*2:2] = r
        # x[:,:,126] = 100
        x[:,:,1:FFT_SIZE*2:2] = i
        # x[:,:,127] = 100
        # x[:,:,:FFT_SIZE*2:2] = iq1.real #in1
        # x[:,:,1:FFT_SIZE*2:2] = iq1.imag #in1

        return x
        # return in1


x1 = torch.randn(1, 1, FFT_SIZE * 2, requires_grad=False,
                dtype=torch.float32)
x2 = torch.randn(1, 1, FFT_SIZE * 2, requires_grad=False,
                dtype=torch.float32)


model = OfdmDivide()
model.eval()

print(x1.shape, model(x1,x2).shape)

scripted = torch.jit.trace(model, [x1, x2])
scripted.save("1/model.pt")
