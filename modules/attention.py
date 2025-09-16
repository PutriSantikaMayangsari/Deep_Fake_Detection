import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalAttention(nn.Module):
    def __init__(self, in_channels=3):
        super(SpatiotemporalAttention, self).__init__()
        
        # Spatial Attention
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sa_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(in_channels)  # apply on channel dimension
        self.relu = nn.ReLU()
        self.sa_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Temporal Attention (linear instead of conv for clarity)
        self.ta_q = nn.Linear(in_channels, in_channels)
        self.ta_k = nn.Linear(in_channels, in_channels)
        self.ta_v = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b*t, c, h, w)

        # --- Spatial Attention ---
        am = self.conv1x1(x_reshaped)  # Eq.(2)
        gf_intermediate = self.sa_conv1(am * x_reshaped)
        gf_intermediate = gf_intermediate.view(b*t, c, -1).transpose(1,2)  # [B*T, H*W, C]
        gf_intermediate = self.layer_norm(gf_intermediate)
        gf_intermediate = gf_intermediate.transpose(1,2).view(b*t, c, h, w)
        gf = self.sa_conv2(self.relu(gf_intermediate))  # Eq.(3)

        saf = x_reshaped + gf  # Eq.(4)

        # --- Temporal Attention ---
        gf_pooled = F.adaptive_avg_pool2d(gf, (1,1)).view(b, t, c)  # [B, T, C]

        q = self.ta_q(gf_pooled)  # [B, T, C]
        k = self.ta_k(gf_pooled)
        v = self.ta_v(gf_pooled)

        attn = torch.bmm(q, k.transpose(1,2)) / (c ** 0.5)  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        tf_matrix = torch.bmm(attn, v)  # [B, T, C]

        tf = tf_matrix.view(b*t, c, 1, 1).expand(-1, -1, h, w)

        # --- Final Output ---
        staf = saf * tf + x_reshaped  # Eq.(6)

        return staf.view(b, t, c, h, w)
