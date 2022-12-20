from torch import nn
import torch

IN_SIZE = 128
OUT_SIZE = 39


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, float_data):
        
        iq = torch.complex(float_data[:,:,::2], float_data[:,:,1::2])
        
        result = iq.view((iq.shape[0], iq.shape[1], OUT_SIZE, IN_SIZE)).permute((0,1,3,2)).flatten(start_dim=2)


        # We do this because TIS doesn't like complex outputs sometimes
        # result = torch.cat([result.real, result.imag], dim=1)
        # return result.permute((0, 2, 1))
        r = result.real
        i = result.imag

        x = torch.zeros(result.shape[0], result.shape[1], result.shape[2]*2, dtype=torch.float32)
        x[:,:,::2] = r
        x[:,:,1::2] = i

        return x


x = torch.randn(1, 1, IN_SIZE*OUT_SIZE * 2, requires_grad=False,
                dtype=torch.float32)


model = Transpose()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
