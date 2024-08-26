import torch
import math
import time
#import pytorch_minimax
#from quantizers import get_transform


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]
Qqs = [torch.tensor(1.0), torch.ones(1)]
Qmax = [1.0, 1.0]


def init(max_bs):
    for i in range(2, max_bs+1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])
        Qqs.append(H)
        Qmax.append(Hmax)


class Preconditioner:
    def __init__(self, x, num_bits, transpose=False, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1
        self.transpose = transpose

        self.x = self.flatten(x)
        #if self.transpose == True:
        #    self.x = torch.t(self.x).contiguous()
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        if self.transpose == True:
            if x.dim() == 4:
                x = x.permute(1, 0, 2, 3).contiguous()
            else:
                x = x.permute(1, 0).contiguous()
        self.x_shape2 = x.shape
        return x.view(x.shape[0], -1)

    def deflatten(self, Tx):
        x = Tx.view(*self.x_shape2)
        if self.transpose == True:
            if x.dim() == 4:
                x = x.permute(1, 0, 2, 3).contiguous()
            else:
                x = x.permute(1, 0).contiguous()
        return x

    def forward(self):
        return self.Tx, self.zero_point, self.scale

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        #if self.transpose == True:
        #    x = torch.t(x).contiguous()
            
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, transpose=False, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, transpose, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        qzero = -self.zero_point * self.scale
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ScalarPreconditionerAct(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, transpose=False, left=True):
        super(ScalarPreconditionerAct, self).__init__(x, num_bits, transpose, left)

    def transform(self, x):
        with torch.no_grad():
            mn = x.min() - 1e-8
            mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ForwardPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, transpose=False, left=True):
        super(ForwardPreconditioner, self).__init__(x, num_bits, transpose, left)

    def transform(self, x):
        with torch.no_grad():
            mn = torch.min(x).mean() - 1e-8
            mx = torch.max(x).mean() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class DiagonalPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, transpose=False, left=True):
        super(DiagonalPreconditioner, self).__init__(x, num_bits, transpose, left)

    def transform(self, x):
        with torch.no_grad():
            if self.left:
                mn = torch.min(x, 1)[0].unsqueeze(1) - 1e-8
                mx = torch.max(x, 1)[0].unsqueeze(1) + 1e-8
                #print(mn.shape)

            else:
                mn = x.min(1)[0] - 1e-8
                mx = x.max(1)[0] + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)
        #print(x.shape, self.zero_point.shape)
        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point



total_time = 0


