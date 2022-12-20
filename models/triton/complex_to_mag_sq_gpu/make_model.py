from torch import nn
import torch

OP_SIZE = 512


class ComplexToMag(nn.Module):
    def __init__(self):
        super(ComplexToMag, self).__init__()

    def forward(self, iq_data ):
        result = \
            torch.add(
                torch.multiply(iq_data[:,:,::2], iq_data[:,:,::2]),
                torch.multiply(iq_data[:,:,1::2], iq_data[:,:,1::2])
            )

        return result


x = torch.randn(1, 1, OP_SIZE*2, requires_grad=False,
                dtype=torch.cfloat)


model = ComplexToMag()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
