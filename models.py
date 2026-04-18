import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_gain=0.02):
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, init_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    net.apply(init_func)


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect', bias=use_bias),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
    ):
        super().__init__()
        assert n_blocks >= 0
        use_bias = True

        layers = [
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=3, padding_mode='reflect', bias=use_bias),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=3, padding_mode='reflect',),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        use_bias = True
        kw, padw = 4, 1

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_res_blocks: int = 9,
        n_disc_layers: int = 3,
    ):
        super().__init__()
        self.G_ab = ResnetGenerator(in_channels, out_channels, ngf=ngf, n_blocks=n_res_blocks)
        self.G_ba = ResnetGenerator(out_channels, in_channels, ngf=ngf, n_blocks=n_res_blocks)
        self.D_a = PatchDiscriminator(in_channels, ndf=ndf, n_layers=n_disc_layers)
        self.D_b = PatchDiscriminator(out_channels, ndf=ndf, n_layers=n_disc_layers)

        for net in (self.G_ab, self.G_ba, self.D_a, self.D_b):
            init_weights(net, init_gain=0.02)

    def forward(self, imgs_a: torch.Tensor, imgs_b: torch.Tensor):
        fake_b = self.G_ab(imgs_a)
        fake_a = self.G_ba(imgs_b)
        rec_a = self.G_ba(fake_b)
        rec_b = self.G_ab(fake_a)
        return fake_a, fake_b, rec_a, rec_b

    def discriminate(
        self,
        imgs_a: torch.Tensor,
        imgs_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ):
        a_real_pred = self.D_a(imgs_a)
        b_real_pred = self.D_b(imgs_b)
        a_fake_pred = self.D_a(fake_a)
        b_fake_pred = self.D_b(fake_b)
        return a_real_pred, b_real_pred, a_fake_pred, b_fake_pred
