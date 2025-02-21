import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor


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
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 5, patch_size = 16, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class PatchUnembedding(nn.Module):
    def __init__(self, in_channels=5, patch_size=16, emb_size=128, img_size=160):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_size = emb_size
        self.in_channels = in_channels
        self.reconstruction = nn.Sequential(
            nn.Linear(emb_size, patch_size * patch_size * in_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=patch_size, p2=patch_size, h=img_size // patch_size, w=img_size // patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.reconstruction(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x, q=None):
        if q is None:
            q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output, q
    

class SelfAttentionMemory(nn.Module):
    def __init__(self, image_size: int, in_channels: int, patch_size: int, emb_size: int) -> None:
        super().__init__()

        self.image_size = image_size
        self.input_dim = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed_h = PatchEmbedding(in_channels, patch_size, emb_size)
        self.patch_embed_m = PatchEmbedding(in_channels, patch_size, emb_size)

        # These are trainable parameters
        self.pos_embedding_h = nn.Parameter(torch.randn(1, self.num_patches, emb_size))
        self.pos_embedding_m = nn.Parameter(torch.randn(1, self.num_patches, emb_size))

        self.attention_h = Attention(emb_size, 4, 0.1)
        self.attention_m = Attention(emb_size, 4, 0.1)

        self.patch_unembed_h = PatchUnembedding(in_channels, patch_size, emb_size, image_size)
        self.patch_unembed_m = PatchUnembedding(in_channels, patch_size, emb_size, image_size)


        # attention for hidden layer
        self.z_h = Conv(in_channels, in_channels, 1, padding="same")
        self.z_m = Conv(in_channels, in_channels, 1, padding="same")

        # weights of concated channels of h Zh and Zm.
        self.w_z = Conv(in_channels * 2, in_channels * 2, 1, padding="same")

        # weights of conated channels of Z and h.
        self.w = Conv(in_channels * 3, in_channels * 3, 1, padding="same")

    def forward(self, h, m):
        """
        Return:
            Tuple(torch.Tensor, torch.Tensor): new Hidden layer and new memory module.
        """

        z_h = self.patch_embed_h(h)
        z_m = self.patch_embed_m(m)

        z_h = z_h + self.pos_embedding_h
        z_m = z_m + self.pos_embedding_m

        z_h, q_h = self.attention_h(z_h)
        z_m, _ = self.attention_m(z_m, q_h)

        z_h = self.patch_unembed_h(z_h)
        z_m = self.patch_unembed_m(z_m)

        z_h = self.z_h(z_h)
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
    def __init__(self, image_size, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.image_size = image_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.W_ci = nn.Parameter(
            torch.zeros(hidden_channels, image_size, image_size, dtype=torch.float32)
            )
        self.W_co = nn.Parameter(
            torch.zeros(hidden_channels, image_size, image_size, dtype=torch.float32)
            )
        self.W_cf = nn.Parameter(
            torch.zeros(hidden_channels, image_size, image_size, dtype=torch.float32)
            )

        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias
                              )
        
        self.attention_memory = SelfAttentionMemory(image_size=image_size, 
                                                    in_channels=hidden_channels, 
                                                    patch_size=16, 
                                                    emb_size=128
                                                    )

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


class SAConvLSTM(nn.Module):
    def __init__(self, image_size, input_channels, output_channels, hidden_channels, kernel_size):
        super(SAConvLSTM, self).__init__()

        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels

        # Create a list of ConvLSTM cells
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(ConvLSTMCell(image_size, input_channels, hidden_channels[0], kernel_size[0]))
        
        # Add subsequent layers
        for i in range(1, self.num_layers):
            self.layers.append(ConvLSTMCell(image_size, hidden_channels[i-1], hidden_channels[i], kernel_size[i]))

        # Bottleneck layer
        self.conv  = nn.Conv2d(hidden_channels[-1], output_channels, kernel_size=1)

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

        return self.conv(h).unsqueeze(dim=1)


if __name__ == "__main__":

    # Test the model
    model = SAConvLSTM(image_size=160, input_channels=5, output_channels=1, hidden_channels=[64, 32, 16], kernel_size=[7, 5, 3])
    x = torch.randn(2, 48, 5, 160, 160)
    out = model(x)
    print(out.shape)
    # Output shape: (2, 1, 1, 160, 160)
    print(model)
