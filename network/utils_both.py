import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


# x -> features
# m -> continuous mask
# s -> error-free signal

class newSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Window_partition(nn.Module):
    def __init__(self, c_in, c_out, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.window_size, self.window_size
        y = x.unfold(2, kH, kH).unfold(3, kW, kW)  # B, C, H, W, kH, kW
        windows = y.permute(0, 1, 4, 5, 2, 3).reshape(B, -1, H // kH, W // kW)
        return windows


class Conv_M(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, padding=0, groups=1, stride=1, bias=True, window_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding, padding, padding)
        self.get_filter1 = nn.Linear((kernel_size ** 2) * c_in * 2, (kernel_size ** 2) * c_in, bias=True)
        self.get_filter2 = nn.Linear((kernel_size ** 2) * c_in * 2, c_in * c_out, bias=True)
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x, m, s):
        # B, C, H, W
        B, C, H, W = m.shape
        with torch.no_grad():
            m_y = F.pad(m, self.padding, mode='replicate')
            m_y = m_y.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
            m_y = m_y.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

            s = F.pad(s, self.padding, mode='replicate')
            s = s.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
            s = s.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

        # B, C, H, W
        y = F.pad(x, self.padding, mode='replicate')
        y = y.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
        y = y.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

        weight = torch.cat([y.reshape(B, H // self.stride, W // self.stride, -1), m_y.reshape(B, H // self.stride, W // self.stride, -1)], -1)
        weight1 = self.get_filter1(weight).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_out, c_in, kh*kw
        weight2 = self.get_filter2(weight).reshape(B, H // self.stride, W // self.stride, self.c_in, self.c_out)

        y = torch.sum(torch.sum(y * weight1, 4, keepdim=True) * weight2, 3).permute(0, 3, 1, 2)

        # mask propagation
        with torch.no_grad():
            m_y = torch.sum(torch.sum(m_y * torch.abs(weight1), 4, keepdim=True) * torch.abs(weight2), 3).permute(0, 3, 1, 2)
            s = torch.sum(torch.sum(s * torch.abs(weight1), 4, keepdim=True) * torch.abs(weight2), 3).permute(0, 3, 1, 2)
            m_y = m_y / s

        return y, m_y, torch.ones_like(m_y)


class ConvT_M(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, padding=0, groups=1, stride=1, bias=True, window_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding, padding, padding)
        self.get_filter1 = nn.Linear((kernel_size ** 2) * c_in * 2, (kernel_size ** 2) * c_in, bias=True)
        self.get_filter2 = nn.Linear((kernel_size ** 2) * c_in * 2, c_in * c_out, bias=True)
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x, m, s):
        # B, C, H, W
        B, C, H, W = m.shape
        with torch.no_grad():
            m_y = F.pad(m, self.padding, mode='replicate')
            m_y = m_y.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
            m_y = m_y.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

            s = F.pad(s, self.padding, mode='replicate')
            s = s.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
            s = s.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

        # B, C, H, W
        y = F.pad(x, self.padding, mode='replicate')
        y = y.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # B, C, H, W, kh, kw
        y = y.permute(0, 2, 3, 1, 4, 5).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_in, kh*kw

        weight = torch.cat([y.reshape(B, H // self.stride, W // self.stride, -1), m_y.reshape(B, H // self.stride, W // self.stride, -1)], -1)
        weight1 = self.get_filter1(weight).reshape(B, H // self.stride, W // self.stride, self.c_in, -1)  # B, H, W, c_out, c_in, kh*kw
        weight2 = self.get_filter2(weight).reshape(B, H // self.stride, W // self.stride, self.c_in, self.c_out)

        y = torch.sum(torch.sum(y * weight1, 4, keepdim=True) * weight2, 3).permute(0, 3, 1, 2)

        # mask propagation
        with torch.no_grad():
            m_y = torch.sum(torch.sum(m_y * torch.abs(weight1), 4, keepdim=True) * torch.abs(weight2), 3).permute(0, 3, 1, 2)
            s = torch.sum(torch.sum(s * torch.abs(weight1), 4, keepdim=True) * torch.abs(weight2), 3).permute(0, 3, 1, 2)
            m_y = m_y / s
            m_y = F.interpolate(m_y, scale_factor=2, mode='bilinear', align_corners=True)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        return y, m_y, torch.ones_like(m_y)


class Linear_M(nn.Module):
    def __init__(self, c_in, c_out, act=None, bias=True):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out, bias=bias)
        self.act = act

    def forward(self, x, m, s):
        x = self.fc(x)
        with torch.no_grad():
            m = m @ torch.abs(self.fc.weight.transpose(1, 0))
            s = s @ torch.abs(self.fc.weight.transpose(1, 0))
            m = m / s
        if self.act is not None:
            x = self.act(x)
        return x, m, torch.ones_like(s)


class LayerNorm_M(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

    def forward(self, x, m, s_):
        # B, H, W, C
        if self.data_format == "channels_last":
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            with torch.no_grad():  # normalize the mask to [0, 1]
                m = m / s_
            return x, m, torch.ones_like(m)
        # B, C, H, W
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
            with torch.no_grad():  # normalize the mask to [0, 1]
                m = m / s_
            return x, m, torch.ones_like(m)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None):
        if mask is None:
            return self.fn(self.norm(x))
        else:
            return self.fn(self.norm(x), mask)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask):
        B = x.shape[0]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        with torch.no_grad():
            updated = (torch.mean(mask, dim=-1, keepdim=True) > 0.) * 1.
            updated = torch.cat([torch.ones_like(updated[:, :1]), updated], 1)
            qkv_m = mask @ torch.abs(self.to_qkv.weight.transpose(1, 0))
            qkv_m = qkv_m.chunk(3, dim=-1)
            q_m, k_m, v_m = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_m)

            q_m = q_m / (1e-6 + torch.max(q_m, dim=2, keepdim=True)[0])
            q_m = torch.cat([torch.ones_like(q_m[:, :, :1]), q_m], 2)

            k_m = k_m / (1e-6 + torch.max(k_m, dim=2, keepdim=True)[0])
            k_m = torch.cat([torch.ones_like(k_m[:, :, :1]), k_m], 2)

            v_m = v_m / (1e-6 + torch.max(v_m, dim=2, keepdim=True)[0])
            v_m = torch.cat([torch.ones_like(v_m[:, :, :1]), v_m], 2)

        dots = torch.matmul(q * q_m, (k * k_m).transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v_m * v)
        out = rearrange(out, 'b h n d -> b n (h d)') * updated

        with torch.no_grad():
            m = torch.matmul(attn, v_m)
            m = rearrange(m, 'b h n d -> b n (h d)') * updated
            m = m @ torch.abs(self.to_out.weight.transpose(1, 0))

        return self.to_out(out), m[:, 1:]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x, m):
        stack = []
        for i, (attn, ff) in enumerate(self.layers):
            y, m = attn(x, m)
            x = y + x
            x = ff(x) + x
        return x, m


# Original version, ref -> https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
