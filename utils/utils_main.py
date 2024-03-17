import numpy as np
import torch
import torch.nn as nn
from utils import utils_image as util




def test_mode(model, L, mode=0, refield=32, min_size=256, sf=1, modulo=1):
    if mode == 0:
        E = test(model, L)
    elif mode == 1:
        E = test_pad(model, L, modulo, sf)
    elif mode == 2:
        E = test_split(model, L, refield, min_size, sf, modulo)
    elif mode == 3:
        E = test_x8(model, L, modulo, sf)
    elif mode == 4:
        E = test_split_x8(model, L, refield, min_size, sf, modulo)
    return E


def test(model, L):
    E = model(L)
    return E


def test_pad(model, L, modulo=16, sf=1):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E,QF = model(L)
    E = E[..., :h*sf, :w*sf]
    return E

def test_pad_deblocking(model, L, modulo=16):
#    embed()
    h, w = L.size()[-2:]
    E0 = model(L)
    paddingH = int(h-np.floor(h/modulo)*modulo)
    paddingW = int(w-np.floor(w/modulo)*modulo)
#    embed()
    top = slice(0,h-paddingH)
    top_c = slice(h-paddingH,h)
    bottom = slice(paddingH,h)
    bottom_c = slice(0,paddingH)
    left = slice(0,w-paddingW)
    left_c = slice(w-paddingW, w)
    right = slice(paddingW,w)
    right_c = slice(0,paddingW)
    L1 = L[...,top,left]
#    L2 = L[...,top,right]
#    embed()
#    L3 = L[...,bottom,left]
#    L4 = L[...,bottom,right]
    E1 = test_mode(model, L1, mode=3)
#    E2 = test_mode(model, L2, mode=3)
    E0[...,top,left] = E1
#    embed()
#    E0[...,top,right] = E2
#    E0[...,top,left_c] = E2[...,top,-1*paddingW:]
#    L1 = torch.nn.ZeroPad2d((0, paddingW, 0, paddingH))(L)
#    L1 = torch.nn.ConstantPad2d((0, paddingW, 0, paddingH),0)(L)
#    E1 = model(L1)[..., :h, :w]
#    L2 = torch.nn.ZeroPad2d((paddingW,0 , 0, paddingH))(L)
#    E2 = model(L2)[..., :h, paddingW:]
#    L3 = torch.nn.ZeroPad2d((0, paddingW, paddingH, 0))(L)
#    E3 = model(L3)[..., paddingH:, :w]
#    L4 = torch.nn.ZeroPad2d((paddingW,0 , paddingH,0))(L)
#    E4 = model(L4)[..., paddingH:, paddingW:]
#    embed()
    E_list = [E0]
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)    
    return E


def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1):
    """
    Args:
        model: trained model
        L: input Low-quality image
        refield: effective receptive filed of the network, 32 is enough
        min_size: min_sizeXmin_size image, e.g., 256X256 image
        sf: scale factor for super-resolution, otherwise 1
        modulo: 1 if split

    Returns:
        E: estimated result
    """
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        L = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
        E = model(L)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i]) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E


def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
    return E



def test_x8(model, L, modulo=1, sf=1):
    E_list = [test_pad(model, util.augment_img_tensor4(L, mode=i), modulo=modulo, sf=sf) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = util.augment_img_tensor4(E_list[i], mode=8 - i)
        else:
            E_list[i] = util.augment_img_tensor4(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E



def test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E_list = [test_split_fn(model, util.augment_img_tensor4(L, mode=i), refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(8)]
    for k, i in enumerate(range(len(E_list))):
        if i==3 or i==5:
            E_list[k] = util.augment_img_tensor4(E_list[k], mode=8-i)
        else:
            E_list[k] = util.augment_img_tensor4(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E



# --------------------------------------------
# print model
# --------------------------------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# --------------------------------------------
# print params
# --------------------------------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


# --------------------------------------------
# model inforation
# --------------------------------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# --------------------------------------------
# params inforation
# --------------------------------------------
def info_params(model):
    msg = describe_params(model)
    return msg

# --------------------------------------------
# model name and total number of parameters
# --------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# --------------------------------------------
# parameters description
# --------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg


if __name__ == '__main__':

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    print_model(model)
    print_params(model)
    x = torch.randn((2,3,401,401))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(5):
            y = test_mode(model, x, mode, refield=32, min_size=256, sf=1, modulo=1)
            print(y.shape)

def batchnormmerge(model):
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        prev_m = m
        batchnormmerge(m)


def SNT(model):
    for k, m in list(model.named_children()):
        if isinstance(m, nn.Sequential):
            if m.__len__() == 1:
                model._modules[k] = m.__getitem__(0)
        SNT(m)

# --------------------------------------------
# SVD Orthogonal Regularization
# --------------------------------------------
def regularizer_orth(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        w = m.weight.data.clone()
        c_out, c_in, f1, f2 = w.size()
        # dtype = m.weight.data.type()
        w = w.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)
        # self.netG.apply(svd_orthogonalization)
        u, s, v = torch.svd(w)
        s[s > 1.5] = s[s > 1.5] - 1e-4
        s[s < 0.5] = s[s < 0.5] + 1e-4
        w = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        m.weight.data = w.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1)  # .type(dtype)
    else:
        pass

def regularizer_clip(m):
    eps = 1e-4
    c_min = -1.5
    c_max = 1.5

    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        w = m.weight.data.clone()
        w[w > c_max] -= eps
        w[w < c_min] += eps
        m.weight.data = w

        if m.bias is not None:
            b = m.bias.data.clone()
            b[b > c_max] -= eps
            b[b < c_min] += eps
            m.bias.data = b