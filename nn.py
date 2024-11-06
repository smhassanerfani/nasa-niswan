from typing import Tuple

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        # Depth-wise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        # Point-wise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SelfAttentionMemory(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        # attention for hidden layer
        self.query_h = Conv(input_dim, hidden_dim, 1, padding="same")
        self.key_h = Conv(input_dim, hidden_dim, 1, padding="same")
        self.value_h = Conv(input_dim, input_dim, 1, padding="same")
        self.z_h = Conv(input_dim, input_dim, 1, padding="same")

        # attention for memory layer
        self.key_m = Conv(input_dim, hidden_dim, 1, padding="same")
        self.value_m = Conv(input_dim, input_dim, 1, padding="same")
        self.z_m = Conv(input_dim, input_dim, 1, padding="same")

        # weights of concated channels of h Zh and Zm.
        self.w_z = Conv(input_dim * 2, input_dim * 2, 1, padding="same")

        # weights of conated channels of Z and h.
        self.w = Conv(input_dim * 3, input_dim * 3, 1, padding="same")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h, m) -> Tuple:
        """
        Return:
            Tuple(torch.Tensor, torch.Tensor): new Hidden layer and new memory module.
        """
        batch_size, _, H, W = h.shape
        # hidden attention
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)

        attention_h = torch.softmax(torch.bmm(q_h, k_h), dim=-1)  # The shape is (batch_size, H*W, H*W)
        z_h = torch.matmul(attention_h, v_h.permute(0, 2, 1))
        z_h = z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_h = self.z_h(z_h)

        # memory attention
        k_m = self.key_m(m)
        v_m = self.value_m(m)

        k_m = k_m.view(batch_size, self.hidden_dim, H * W)
        v_m = v_m.view(batch_size, self.input_dim, H * W)

        attention_m = torch.softmax(torch.bmm(q_h, k_m), dim=-1)
        z_m = torch.matmul(attention_m, v_m.permute(0, 2, 1))
        z_m = z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        z_m = self.z_m(z_m)

        # channel concat of Zh and Zm.
        Z = torch.cat([z_h, z_m], dim=1)
        Z = self.w_z(Z)

        # channel concat of Z and h
        W = torch.cat([Z, h], dim=1)
        W = self.w(W)

        # mi_conv: Wm; zi * Z + Wm; hi * Ht + bm; i
        # mg_conv: Wm; zg * Z + Wm; hg * Ht + bm; g
        # mo_conv: Wm; zo * Z + Wm; ho * Ht + bm; o
        mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=3, dim=1)
        input_gate = torch.sigmoid(mi_conv)
        g = torch.tanh(mg_conv)
        new_M = (1 - input_gate) * m + input_gate * g
        output_gate = torch.sigmoid(mo_conv)
        new_H = output_gate * new_M

        return new_H, new_M


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.W_ci = nn.parameter.Parameter(
            torch.zeros(hidden_channels, 32, 32, dtype=torch.float)
        )
        self.W_co = nn.parameter.Parameter(
            torch.zeros(hidden_channels, 32, 32, dtype=torch.float)
        )
        self.W_cf = nn.parameter.Parameter(
            torch.zeros(hidden_channels, 32, 32, dtype=torch.float)
        )

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.attention_memory = SelfAttentionMemory(hidden_channels, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state):
        h, c, m = hidden_state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate  = torch.split(gates, self.hidden_channels, dim=1)
        
        ingate     = self.sigmoid(ingate + self.W_ci * c)
        forgetgate = self.sigmoid(forgetgate + self.W_cf * c)
        cellgate   = self.tanh(cellgate)
        # outgate    = self.sigmoid(outgate)

        c = c * forgetgate + ingate * cellgate
        
        outgate = self.sigmoid(outgate + self.W_co * c)
        h = outgate * self.tanh(c)
        h, m = self.attention_memory(h, m)
        return h, c, m


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        assert len(hidden_channels) == num_layers, 'The length of hidden_channels must be equal to num_layers.'
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

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
            h = torch.zeros(batch_size, self.hidden_channels[i], height, width).to(x.device)
            c = torch.zeros(batch_size, self.hidden_channels[i], height, width).to(x.device)
            m = torch.zeros(batch_size, self.hidden_channels[i], height, width).to(x.device)
            hidden_states.append((h, c, m))

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h, c, m = hidden_states[layer_idx]
                h, c, m = self.layers[layer_idx](x_t, (h, c, m))
                hidden_states[layer_idx] = (h, c, m)
                x_t = h  # Output of the current layer is the input to the next layer

        return self.conv(h).unsqueeze(1) # torch.cat(outputs, dim=1)


def main():

    x = torch.randn((2, 48, 5, 32, 32))
    model = ConvLSTM(5, [64, 32, 16], [5, 3, 3], 3)
    
    print(model(x).shape) # torch.Size([2, 1, 1, 32, 32])

if __name__ == '__main__':
    main()