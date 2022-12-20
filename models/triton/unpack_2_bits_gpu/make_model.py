from torch import nn
import torch

OP_SIZE = 512


class Unpack2Bits(nn.Module):
    def __init__(self):
        super(Unpack2Bits, self).__init__()

    def forward(self, inp ):

        result = torch.zeros(inp.shape[0], inp.shape[1], inp.shape[2]*2, dtype=torch.int8)
        # Grab the MSB
        result[:,:,::2] = torch.bitwise_right_shift(torch.bitwise_and(inp, 0x02), 1)
        # Grab the LSB
        result[:,:,1::2] = torch.bitwise_and(inp, 0x01)

        return result


x = torch.randint(4, (1, 1, OP_SIZE), requires_grad=False,
                dtype=torch.int8)


model = Unpack2Bits()
model.eval()

print(x.shape, model(x).shape, model(x).dtype )

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
