import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = conv_block(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=186, out_channels=20):
        super(UNet, self).__init__()

        self.bottleneck1 = nn.Conv2d(in_channels, 64, 1)

        self.enc1 = Encoder(64, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        self.bottleneck2 = conv_block(512, 1024)

        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, e, f):

        B, C1, _, H, W = e.size()
        B, C2, L, H, W = f.size()

        e = e.view(B, C1, H, W)    # [96, 6, 256, 256]
        f = f.view(B, C2*L, H, W)  # [96, 180, 256, 256]
        
        x = torch.cat([e, f], dim=1)
        x = self.bottleneck1(x)  # [96, 186, 256, 256]

        # Encoder path
        x1, x1_pooled = self.enc1(x)
        x2, x2_pooled = self.enc2(x1_pooled)
        x3, x3_pooled = self.enc3(x2_pooled)
        x4, x4_pooled = self.enc4(x3_pooled)

        # Bottleneck
        x_bottleneck = self.bottleneck2(x4_pooled)

        # Decoder path
        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        # Final convolution
        x = self.final_conv(x)
        return x.unsqueeze(dim=1)
    

if __name__ == "__main__":

    e = torch.randn((2, 6, 1, 256, 256))
    f = torch.randn((2, 9, 20, 256, 256))

    m = UNet(in_channels=6+180, out_channels=20)

    y = m(e, f)
    print(y.shape) # torch.Size([2, 1, 1, 20, 256, 256])

  