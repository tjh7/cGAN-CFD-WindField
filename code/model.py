
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if use_act else nn.Identity(),
        )
   
    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, condition_dim=4):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, 256 * 256)

        self.down1 = UNetBlock(in_channels + 1, 64, down=True, kernel_size=4, stride=2, padding=1)
        self.down2 = UNetBlock(64, 128, down=True, kernel_size=4, stride=2, padding=1)
        self.down3 = UNetBlock(128, 256, down=True, kernel_size=4, stride=2, padding=1)
        self.down4 = UNetBlock(256, 512, down=True, kernel_size=4, stride=2, padding=1)
        self.up1 = UNetBlock(512, 256, down=False, kernel_size=4, stride=2, padding=1)
        self.up2 = UNetBlock(512, 128, down=False, kernel_size=4, stride=2, padding=1)
        self.up3 = UNetBlock(256, 64, down=False, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, condition):
        cond_map = self.condition_proj(condition).view(-1, 1, 256, 256)
        x = torch.cat([x, cond_map], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        out = self.up4(torch.cat([u3, d1], dim=1))
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):  
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),  # 256->128
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),          # 128->64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),         # 64->32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 1, 1),         # 32->31
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, 1, 1),           # 31->30

        )

    def forward(self, x):
        return self.model(x).view(-1)
