import torch
import torch.nn as nn


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
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_channels)

        # Create a list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(ConvLSTMCell(in_channels, hidden_channels[0], kernel_size[0]))
        
        # Add subsequent layers
        for i in range(1, self.num_layers):
            self.layers.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size[i]))

        # Bottleneck layer
        self.conv  = nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=1)

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
            # outputs.append(self.conv(h)) 

        return self.conv(h).unsqueeze(dim=1) # , torch.cat(outputs, dim=1)


if __name__ == '__main__':
    x = torch.randn((2, 48, 5, 32, 32))

    model = ConvLSTM(5, 1, [64, 32, 16], [7, 5, 3])
    print(model(x).shape)