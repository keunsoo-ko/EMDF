from .utils_both import *


class newSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ViT(nn.Module):
    def __init__(self, num_token, dim, depth=4, heads=8, mlp_dim=1024, dim_head=64):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_token + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, x, m):
        B, C, H, W = x.shape
        # B, C, H, W -> B, N(HW), C
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        m = m.permute(0, 2, 3, 1).reshape(B, H * W, C)

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        x, m = self.attn(x, m)
        # B, N(HW), C -> B, C, H, W
        x = x[:, 1:].reshape(B, H, W, C).permute(0, 3, 1, 2)
        m = m.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x, m


class Block(nn.Module):  # ConvNeXtv2
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = Conv_M(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)  # depthwise conv
        self.norm = LayerNorm_M(dim)
        self.pwconv1 = Linear_M(dim, 4 * dim, bias=False)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = Linear_M(4 * dim, dim)

    def forward(self, x, m, s):
        res = x
        # res_m = m
        x, m, s = self.dwconv(x, m, s)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        m = m.permute(0, 2, 3, 1)
        s = s.permute(0, 2, 3, 1)
        x, m, s = self.norm(x, m, s)
        x, m, s = self.pwconv1(x, m, s)
        x = self.grn(self.act(x))
        x, m, s = self.pwconv2(x, m, s)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        m = m.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        s = s.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # for residual, assume that error-free + error -> error-free
        output = x + res
        with torch.no_grad():
            output_m = m / s
            # output_m = (output_m + res_m).clamp(0, 1)
        return output, output_m, s


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=4, num_classes=3, depths=[3, 3, 5, 3], dims=[16, 32, 64, 128], k_size=[7, 7, 5, 3], head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = newSequential(
            Conv_M(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            LayerNorm_M(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = newSequential(
                LayerNorm_M(dims[i], eps=1e-6, data_format="channels_first"),
                Conv_M(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stage = newSequential(*[Block(dim=dims[i], kernel_size=k_size[i]) for j in range(depths[i])])
            self.stages.append(stage)

        # Decoder
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            upsample_layer = newSequential(
                LayerNorm_M(dims[3 - i], eps=1e-6, data_format="channels_first"),
                ConvT_M(dims[3 - i], dims[2 - i]),
            )
            self.upsample_layers.append(upsample_layer)
        self.de_stages = nn.ModuleList()
        for i in range(3):
            stage = newSequential(*[Block(dim=dims[2 - i], kernel_size=k_size[2 - i]) for j in range(depths[2 - i])])
            self.de_stages.append(stage)

        self.head = newSequential(
            LayerNorm_M(dims[0], data_format="channels_first"),
            Conv_M(dims[0], 4 * 4 * 3, kernel_size=1))

        self.out = nn.Sequential(
            nn.PixelShuffle(4),
            nn.Tanh()
        )

    def forward_features(self, x, m):
        s = torch.ones_like(m)
        skip = []
        m_ = []
        for i in range(4):
            x, m, s = self.downsample_layers[i](x, m, s)
            x, m, s = self.stages[i](x, m, s)
            skip.append(x)
            m_.append(m)
        skip.pop()
        # decoder
        for i in range(3):
            x, m, s = self.upsample_layers[i](x, m, s)
            x = x + skip[-1]
            x, m, s = self.de_stages[i](x, m, s)
            m_.append(m)
            skip.pop()
        out, _, _ = self.head(x, m, s)
        return self.out(out), m_

    def forward(self, img, mask):
        m = mask.repeat(1, 4, 1, 1)
        x = torch.cat((img, mask), 1)
        gen, m_ = self.forward_features(x, m)

        return gen, m_

    # def forward(self, img, mask, calculate_loss=True):
    #     if calculate_loss:
    #         m = (1 - mask).repeat(1, 4, 1, 1)
    #         m[:, :2] = 1
    #         x = torch.cat((img * m[:, :3], mask), 1)
    #     else:
    #         m = (1 - mask).repeat(1, 4, 1, 1)
    #         x = torch.cat((img * (1 - mask), mask), 1)
    #     out, _ = self.forward_features(x, m)
    #     return out, 1 - m[:, :3]
