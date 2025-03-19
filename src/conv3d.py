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
        # x = F.pad(x, (0, 0, 0, 0, self.causal_padding, 0, 0, 0, 0, 0)) # Pad the depth dimension

        gates = self.conv(x)
        x, outgate  = torch.split(gates, self.out_channels, dim=1)

        return x * self.sigmoid(outgate)


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 1, 1)):
        super(Conv3D, self).__init__()

        # Explicitly define each Conv3D layer with correct dilation
        self.conv1 = GLU(
            256,
            256,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(1, 1, 1)
        )

        self.conv2 = GLU(
            256,
            128,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(2, 1, 1)
        )

        self.conv3 = GLU(
            128,
            128,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(4, 1, 1)
        )

        self.conv4 = GLU(
            128,
            64,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(8, 1, 1)
        )

        self.conv5 = GLU(
            64,
            22,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(16, 1, 1)
        )

        # self.conv6 = GLU(
        #     64,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     dilation=(32, 1, 1)
        # )
       
        self.encoder = Encoder(in_channels, 5)
        # self.decoder = Decoder(1, 5)
        
        # self.readout_ftr = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)
        # self.readout_tmp = nn.Conv2d(48, 1, kernel_size=1, stride=1)
        self.readout = nn.Conv2d(22, out_channels, kernel_size=1, stride=1)

    def forward(self, x, e):
        B, T, C, L, H, W = x.size() # torch.Size([8, 48, 4, 22, 90, 144]) 

        x = x.view(B*T, C*L, H, W) # torch.Size([384, 88, 90, 144])
        e = e.view(B*T, 1, H, W)   # torch.Size([384, 1, 90, 144])
        x = torch.concat((x, e), dim=1) # torch.Size([384, 89, 90, 144])

        x, _ = self.encoder(x) # BxT, 256, H, W

        x = x.view(B, T, 256, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # BxTxFxHxW -> BxFxTxHxW

        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.conv5(x) 
        # x = self.conv6(x) # Bx22x48xHxW
        x = x.view(B, L, H, W)
        x = self.readout(x)

        # x = x.view(B*L, T, H, W)
        # x = self.readout_tmp(x)
        x = x.view(B, 1, 1, L, H, W) # torch.Size([8, 1, 1, 22, 90, 144])

        return x

class Encoder(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(in_channels,  128, kernel_size, stride=1)
        self.conv2 = ConvLayer(128, 128, kernel_size, stride=1)
        self.conv3 = ConvLayer(128, 256, kernel_size, stride=1)
        self.conv4 = ConvLayer(256, 256, kernel_size, stride=1)
    
    def forward(self, x):  # BxT, 5, 160, 160
        enc = self.conv1(x)
        x = self.conv2(enc)
        x = self.conv3(x)
        x = self.conv4(x)

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

    x = torch.randn((2, 32, 4, 22, 90, 144))
    e = torch.randn((2, 32, 1, 1, 90, 144))
    model = Conv3D(in_channels=89, out_channels=22, kernel_size=(2, 3, 3))
    y = model(x, e)
    print(y.size())
    print(model)

if __name__ == '__main__':
    main()