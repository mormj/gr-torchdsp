from torch import nn
import torch

OP_SIZE = 512


class Log10(nn.Module):
    def __init__(self):
        super(Log10, self).__init__()

    def forward(self, float_data ):
        return torch.log10(float_data)


x = torch.randn(1, 1, OP_SIZE, requires_grad=False,
                dtype=torch.float)


model = Log10()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")