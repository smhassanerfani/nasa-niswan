import torch
import torch.nn as nn

def _apply_padding(data, padding):
    """
    Apply cyclic padding along the longitude (W dimension)
    and reflective padding along the latitude (H dimension).
    """
    pad_lat = padding[0]  # top and bottom
    pad_lon = padding[1]  # left and right

    H = data.shape[-2]  # latitude
    W = data.shape[-1]  # longitude

    # --- Cyclic padding along longitude (axis=4) ---
    if (pad_lon <= W):
        data = torch.concatenate([data[..., -pad_lon:], data, data[..., :pad_lon]], axis=-1)
    else:
        raise AttributeError(
            f"Requested longitude padding ({pad_lon}) exceeds image width ({W})."
        )

    # --- Reflective padding along latitude (axis=3) ---
    pad_top = pad_lat
    pad_bottom = pad_lat

    if (pad_top <= H) and (pad_bottom <= H):
        top = torch.flip(data[..., 1:pad_top+1, :], dims=[-2])
        bottom = torch.flip(data[..., -pad_bottom-1:-1, :], dims=[-2])
        data = torch.concatenate((top, data, bottom), axis=-2)
    else:
        raise AttributeError(
            f"Requested latitude padding ({pad_top}, {pad_bottom}) exceeds image height ({H})."
        )

    return data


class LatitudeWeightedMSE(nn.Module):
    def __init__(self, lat_res=90, device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. Generate Latitude weights (Cosine of latitude)
        # Grid assumes typically [90, -90] or [-90, 90]. 
        # For Equiangular grid, we create a 1D tensor of weights.
        
        # Create a linear space for latitude from -90 to 90 (in radians)
        lats = torch.linspace(-torch.pi/2, torch.pi/2, lat_res)
        
        # Calculate cosine weights
        weights = torch.cos(lats)
        
        # Reshape to broadcast: (1, 1, Lat, 1) -> fits (B, C, H, W)
        # We assume H=lat_res
        self.weights = weights.view(1, 1, -1, 1).to(device)

    def forward(self, pred, target):
        # Squared Error
        diff_sq = (pred - target) ** 2
        
        # Weighted Mean
        # We multiply error by weights, then mean.
        weighted_mse = torch.mean(diff_sq * self.weights)
        
        return weighted_mse
