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
            in_channels,
            16,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(1, 1, 1)
        )

        self.conv2 = GLU(
            16,
            32,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(2, 1, 1)
        )

        self.conv3 = GLU(
            32,
            32,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(4, 1, 1)
        )

        self.conv4 = GLU(
            32,
            32,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(8, 1, 1)
        )

        self.conv5 = GLU(
            32,
            16,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(16, 1, 1)
        )

        self.conv6 = GLU(
            16,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=(32, 1, 1)
        )
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state):
        h, c = hidden_state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate  = torch.split(gates, self.hidden_channels, dim=1)
        
        ingate     = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate   = self.tanh(cellgate)
        outgate    = self.sigmoid(outgate)

        c = c * forgetgate + ingate * cellgate
        h = outgate * self.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_channels)

        # Create a list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size[0]))
        
        # Add subsequent layers
        for i in range(1, self.num_layers):
            self.layers.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size[i]))

        # Bottleneck layer
        self.conv  = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        # Assuming x is a sequence of frames: (batch_size, sequence_length, channels, height, width)
        batch_size, _, seq_len, height, width = x.size()
        
        # Initialize hidden and cell states for each layer
        hidden_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            c = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            hidden_states.append((h, c))

        for t in range(seq_len):
            x_t = x[:, :, t, :, :]
            for layer_idx in range(self.num_layers):
                h, c = hidden_states[layer_idx]
                h, c = self.layers[layer_idx](x_t, (h, c))
                hidden_states[layer_idx] = (h, c)
                x_t = h  # Output of the current layer is the input to the next layer

        return self.conv(h).unsqueeze(1)


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
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(in_channels,  out_channels, kernel_size, stride=1)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride=2)
        self.conv3 = ConvLayer(out_channels, out_channels, kernel_size, stride=1)
        self.conv4 = ConvLayer(out_channels, out_channels, kernel_size, stride=2)
    
    def forward(self, x):  # BxT, 5, 160, 160
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Decoder(nn.Module):

    def __init__(self, hid_channels, out_channels, kernel_size):
        super(Decoder, self).__init__()
        self.conv1 = ConvTransposeLayer(hid_channels, hid_channels, kernel_size, stride=1)
        self.conv2 = ConvLayer(hid_channels, hid_channels, kernel_size, stride=1)
        self.conv3 = ConvTransposeLayer(hid_channels, hid_channels, kernel_size, stride=1)
        self.conv4 = ConvLayer(hid_channels, hid_channels, kernel_size, stride=1)
        self.readout = nn.Conv2d(hid_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):  # BxT, 5, 40, 40
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.readout(x)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_channels)

        # Create a list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size[0]))
        
        # Add subsequent layers
        for i in range(1, self.num_layers):
            self.layers.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size[i]))

        # Bottleneck layer
        # self.conv  = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        # Assuming x is a sequence of frames: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden and cell states for each layer
        hidden_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            c = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            hidden_states.append((h, c))

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h, c = hidden_states[layer_idx]
                h, c = self.layers[layer_idx](x_t, (h, c))
                hidden_states[layer_idx] = (h, c)
                x_t = h  # Output of the current layer is the input to the next layer  
        return h.unsqueeze(1)


class EncDecConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(EncDecConvLSTM, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, kernel_size)
        self.decoder = Decoder(hidden_channels, 1, kernel_size)
        self.lstm = ConvLSTM(hidden_channels, [hidden_channels, hidden_channels, hidden_channels], [5, 3, 3])

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)

        z = self.encoder(x)

        C_new, H_new, W_new = z.size(1), z.size(2), z.size(3)
        z = z.view(B, T, C_new, H_new, W_new)

        z = self.lstm(z)

        T_new = z.size(1)
        z = z.view(B*T_new, C_new, H_new, W_new)

        out = self.decoder(z)
        out = out.view(B, T_new, 1, H, W) 
        return out


def main():

    x = torch.randn((8, 5, 64, 100, 154))
    model = ConvLSTM(5, [64, 32, 16], [5, 3, 3])
    # model = Conv3D(in_channels=5, out_channels=1, kernel_size=(2, 3, 3))
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    main()