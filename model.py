import torch
import torch.nn as nn


class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super(DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        out = self.initial_layer(torch.concat([x, y], dim=1))
        return self.model(out)


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, encoder=True, act='ReLU', use_dropout=False):
        super(GBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if encoder
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == 'ReLU' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):

    def __init__(self, in_channels, features=64):
        super(Generator, self).__init__()
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encode1 = GBlock(features,   features*2, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode2 = GBlock(features*2, features*4, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode3 = GBlock(features*4, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode4 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode5 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode6 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            # nn.ReLU()
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decode1 = GBlock(features*8,   features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode2 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode3 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode4 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=False)
        self.decode5 = GBlock(features*8*2, features*4, encoder=False, act='ReLU', use_dropout=False)
        self.decode6 = GBlock(features*4*2, features*2, encoder=False, act='ReLU', use_dropout=False)
        self.decode7 = GBlock(features*2*2, features, encoder=False, act='ReLU', use_dropout=False)
        
        self.final_decode = nn.Sequential(
            nn.ConvTranspose2d(features*2, 1, kernel_size=4, stride=2, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        en0 = self.initial_encoder(x)
        en1 = self.encode1(en0)
        en2 = self.encode2(en1)
        en3 = self.encode3(en2)
        en4 = self.encode4(en3)
        en5 = self.encode5(en4)
        en6 = self.encode6(en5)
        bn = self.bottleneck(en6)
        de1 = self.decode1(bn)
        de2 = self.decode2(torch.cat([de1, en6], dim=1))
        de3 = self.decode3(torch.cat([de2, en5], dim=1))
        de4 = self.decode4(torch.cat([de3, en4], dim=1))
        de5 = self.decode5(torch.cat([de4, en3], dim=1))
        de6 = self.decode6(torch.cat([de5, en2], dim=1))
        de7 = self.decode7(torch.cat([de6, en1], dim=1))
        return self.final_decode(torch.cat([de7, en0], dim=1))


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
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = Encoder(in_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        self.bottleneck = conv_block(512, 1024)

        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1, x1_pooled = self.enc1(x)
        x2, x2_pooled = self.enc2(x1_pooled)
        x3, x3_pooled = self.enc3(x2_pooled)
        x4, x4_pooled = self.enc4(x3_pooled)

        # Bottleneck
        x_bottleneck = self.bottleneck(x4_pooled)

        # Decoder path
        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        # Final convolution
        x = self.final_conv(x)
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
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        assert len(hidden_channels) == num_layers, 'The length of hidden_channels must be equal to num_layers.'
        self.num_layers = num_layers

        # Create a list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size[0]))
        
        # Add subsequent layers
        for i in range(1, num_layers):
            self.layers.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size[i]))

        # Bottleneck layer
        self.conv  = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        # Assuming x is a sequence of frames: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden and cell states for each layer
        hidden_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            c = torch.zeros(batch_size, self.layers[i].hidden_channels, height, width).to(x.device)
            hidden_states.append((h, c))

        # outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h, c = hidden_states[layer_idx]
                h, c = self.layers[layer_idx](x_t, (h, c))
                hidden_states[layer_idx] = (h, c)
                x_t = h  # Output of the current layer is the input to the next layer
            # outputs.append(h.unsqueeze(1)) 

        return self.conv(h) # torch.cat(outputs, dim=1)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    x = torch.randn((2, 48, 5, 100, 154))
    # y = torch.randn((1, 1, 256, 256))
    # disc = Discriminator(in_channels=1)
    # print(disc(x, y).shape)

    # model = UNet(in_channels=5, out_channels=1)
    # print(model(x).shape)

    model = ConvLSTM(5, [64, 32, 16], [5, 3, 3], 3)
    print(model(x).shape)

if __name__ == '__main__':
    main()