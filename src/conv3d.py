import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(GLU, self).__init__()
        self.out_channels = out_channels
        self.causal_padding  = dilation[0] * (kernel_size[0] - 1)  # Causal padding for the temporal dimension
        
        self.conv = nn.Conv3d(
            in_channels, out_channels*2, kernel_size, padding=padding, dilation=dilation
        ) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, self.causal_padding, 0, 0, 0, 0, 0)) # Pad the depth dimension

        gates = self.conv(x)
        x, outgate  = torch.split(gates, self.out_channels, dim=1)

        return x * self.sigmoid(outgate)


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 1, 1)):
        super(Conv3D, self).__init__()

        # Explicitly define each Conv3D layer with correct dilation
        self.conv1 = GLU(
            64,
            64,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(1, 1, 1)
        )

        self.conv2 = GLU(
            64,
            32,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(2, 1, 1)
        )

        self.conv3 = GLU(
            32,
            16,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(4, 1, 1)
        )

        self.conv4 = GLU(
            16,
            8,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(8, 1, 1)
        )

        self.conv5 = GLU(
            8,
            4,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(16, 1, 1)
        )

        self.conv6 = GLU(
            4,
            1,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(32, 1, 1)
        )
       
        self.encoder = Encoder(in_channels, 5)
        # self.decoder = Decoder(1, 5)
        
        # self.readout_ftr = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)
        self.readout_tmp = nn.Conv2d(48, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        # H_enc, W_enc = H//2, W//2

        x = x.view(B*T, C, H, W)
        x, _ = self.encoder(x) # BxT, F, H//2, W//2

        x = x.view(B, T, 64, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # BxTxFxHxW -> BxFxTxHxW

        x = self.conv1(x) # 128
        x = self.conv2(x) # 64
        x = self.conv3(x) # 32
        x = self.conv4(x) # 16
        x = self.conv5(x) # 8
        x = self.conv6(x) # 1 -> Bx1x1xHxW

        # x = x.permute(0, 2, 1, 3, 4)  # BxFxCxHxW -> BxTxFxHxW
        # x = x.contiguous().view(B*T, 1, H_enc, W_enc)

        # x = self.decoder(x) # BxT, 1, 32, 32

        x = x.view(B, T, H, W)
        x = self.readout_tmp(x)

        return x.unsqueeze(2)  # BxT, 1, 1, 32, 32

class Encoder(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(in_channels,  64, kernel_size, stride=1)
        self.conv2 = ConvLayer(64, 64, kernel_size, stride=1)
        self.conv3 = ConvLayer(64, 64, kernel_size, stride=1)
        self.conv4 = ConvLayer(64, 64, kernel_size, stride=1)
    
    def forward(self, x):  # BxT, 5, 160, 160
        enc = self.conv1(x)
        x = self.conv2(enc)

        return x, enc


class Decoder(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(Decoder, self).__init__()
        self.conv1 = ConvTransposeLayer(in_channels, 1, kernel_size, stride=1)
        self.conv2 = ConvLayer(1, 1, kernel_size, stride=1)

    def forward(self, x, enc=None):  # BxT, 1, 16, 16
        x = self.conv1(x)
        if enc is not None:
            x = self.conv2(x + enc)
        else:
            x = self.conv2(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - stride + 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding
        )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTransposeLayer, self).__init__()
        padding = (kernel_size - stride + 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                      stride=1, padding=padding),
            nn.PixelShuffle(2)
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
def main():

    x = torch.randn((8, 48, 5, 90, 144))
    model = Conv3D(in_channels=5, out_channels=1, kernel_size=(2, 3, 3))
    y = model(x)
    print(y.size())
    print(model)

if __name__ == '__main__':
    main()