import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange


class Patcher(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size, padding=0, count_include_pad=False)

    def forward(self, x):
        return self.pool(x).flatten(2).transpose(1, 2)


class MaskerModel(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature = vgg19.features
        self.img = Patcher(patch_size=patch_size)

    def forward(self, x):
        return self.feature(x), self.img(x)


hyper_t = 1


def token_variance_loss(x, mask):  # B N C, B K N
    B, N, C = x.size()
    B, K, N = mask.size()
    # print('B, N, C', B, N, C)
    # print('B, K, N', B, K, N)

    z = x.new().resize_(K, B, C).fill_(0)
    l = x.new().resize_(K, B).fill_(0)

    mask = rearrange(mask, 'B K N -> K B N')
    for k in range(0, K):
        # print('[ym]-----------------torch.norm(x)', torch.norm(x, dim=2))
        masked_x = x*(mask[k].unsqueeze(2))  # B N C
        # print('[ym]-----------------torch.norm(masked_x, dim=2)', torch.norm(masked_x, dim=2))
        sumed_masked_x = masked_x.sum(dim=1)  # B C
        # print('[ym]-----------------sumed_masked_x', sumed_masked_x)
        sumed_mask = mask[k].sum(dim=1)  # B
        # print('[ym]-----------------sumed_mask', sumed_mask)
        zb = sumed_masked_x / (sumed_mask.unsqueeze(1))  # B C
        zb = torch.where(torch.isnan(zb), torch.full_like(zb, 0), zb)  # some mask is empty, cause sumed_mask is zero
        # print('[ym]-----------------torch.norm(zb)', torch.norm(zb, dim=1))
        lb = (torch.norm(x - zb.unsqueeze(dim=1), dim=2)*mask[k]).sum(dim=1)  # B N C -> B N -> B
        # print('[ym]-----------------lb', lb)
        z[k] = zb
        l[k] = lb
        if torch.any(torch.isnan(lb)):
            print('[ym]-----------------x', x)
            print('[ym]-----------------torch.norm(x)', torch.norm(x, dim=2))
            print('[ym]-----------------mask.sum(dim=1)', mask.sum(dim=1))
            print('[ym]-----------------mask', mask)
            print('[ym]-----------------mask[k]', mask[k])
            print('[ym]-----------------torch.norm(masked_x, dim=2)', torch.norm(masked_x, dim=2))
            print('[ym]-----------------sumed_masked_x', sumed_masked_x)
            print('[ym]-----------------sumed_mask', sumed_mask)
            print('[ym]-----------------torch.norm(zb)', torch.norm(zb, dim=1))

    return z, l  # K B C, K B


def token_contrastive_loss(z):  # K B C
    K, B, C = z.size()
    # print('K, B, C', K, B, C)
    # print('[ym]-----------------z', z)

    zkb = rearrange(z, 'K B C -> (K B) C')

    zkb_dot = zkb @ zkb.t()  # K*B K*B
    zkb_exp = torch.exp(zkb_dot/hyper_t)
    # print('[ym]-----------------zkb', zkb)
    # print('[ym]-----------------zkb.norm(zkb, dim=1)', torch.norm(zkb, dim=1))
    # print('[ym]-----------------zkb_dot', zkb_dot)
    # print('[ym]-----------------zkb_exp', zkb_exp)

    zkb_mask_p = zkb_dot.new().resize_(K * B, K * B).fill_(0)  # K*B K*B
    for k in range(0, K):
        src = torch.arange(k*B, k*B+B)
        dst = torch.randint(k*B, k*B+B, (B,))
        zkb_mask_p[src, dst] = 1  # FIXME

    zkb_mask_n = zkb_dot.new().resize_(K*B, K*B).fill_(0)  # K*B K*B
    for k in range(0, K):
        zkb_mask_n[k*B:k*B+B, k*B:k*B+B] = 1  # FIXME
    zkb_mask = torch.where(zkb_mask_n == 0, 1, 0)

    positives = (zkb_exp*zkb_mask_p).sum(dim=1).reshape(K, B)  # K*B K*B -> K*B -> K B
    negatives = (zkb_exp*zkb_mask).sum(dim=1).reshape(K, B)  # K*B K*B -> K*B -> K B
    # print('[ym]-----------------positives', positives)
    # print('[ym]-----------------negatives', negatives)
    l = -torch.log(positives/(positives+negatives))  # K B
    # print('[ym]-----------------l', l)

    if torch.any(torch.isnan(l)):
        print('[ym]-----------------zkb', zkb)
        print('[ym]-----------------zkb.norm(zkb, dim=1)', torch.norm(zkb, dim=1))
        print('[ym]-----------------zkb_dot', zkb_dot)
        print('[ym]-----------------zkb_exp', zkb_exp)
        print('[ym]-----------------positives', positives)
        print('[ym]-----------------negatives', negatives)

    return l


def part_loss(x, img, mask):  # B N C, B N C, B K N
    x = F.normalize(x, dim=2)  # must normalize
    # print('[ym]-----------------x', x)
    # print('[ym]-----------------torch.norm(x, dim=2)', torch.norm(x, dim=2))
    # print('[ym]-----------------img', img)
    # print('[ym]-----------------torch.norm(img, dim=2)', torch.norm(img, dim=2))

    # print('[ym]-----------------mask.sum(dim=1)', mask.sum(dim=1))
    mask = nn.Softmax(dim=1)(mask)
    # print('[ym]-----------------Softmax mask.sum(dim=1)', mask.sum(dim=1))

    feature_z, feature_vl = token_variance_loss(x, mask)
    feature_cl = token_contrastive_loss(feature_z)
    # image_z, image_vl = token_variance_loss(img, mask)

    # print('[ym]-----------------feature_vl', feature_vl.sum())
    # print('[ym]-----------------feature_cl', feature_cl.sum())
    # print('[ym]-----------------image_vl', image_vl.sum())

    # return (feature_vl + feature_cl + image_vl).sum()
    return (feature_vl + feature_cl).sum()
