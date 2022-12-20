from torch import nn
import torch

OP_SIZE = 512


class QPSKDecoder(nn.Module):
    def __init__(self):
        super(QPSKDecoder, self).__init__()

    def forward(self, iq_data ):

        # // Real component determines small bit.
        # // Imag component determines big bit.
        # return 2 * (imag(*sample) > 0) + (real(*sample) > 0);
        
        r = iq_data[:,:,::2]
        i = iq_data[:,:,1::2]


        result = (torch.add(2.0 * torch.where(i > 0, 1.0, 0.0),torch.where(r > 0, 1.0, 0.0))).type(dtype=torch.int8)
        
        return result


x = torch.randn(1, 1, OP_SIZE*2, requires_grad=False,
                dtype=torch.float)


model = QPSKDecoder()
model.eval()

print(x.shape, model(x).shape, model(x).dtype )

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
