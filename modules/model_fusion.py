# modules/model_fusion.py
# Model dengan dua cabang: LBP + STA, masing-masing pakai Xception, 
# lalu digabung (concat/add) dan dimodelkan temporalnya dengan LSTM.
# LBP fokus pada tekstur, STA pada perubahan spasial temporal.
# mantap juga nih idenya, tunggu hasilnya nanti gess

"""
Bisa diuji dua mode:

fusion="concat" → lebih informatif, tapi param lebih besar.

fusion="add" → lebih ringan.
"""

import torch
import torch.nn as nn
import timm
from modules.attention import SpatiotemporalAttention
from config import CONFIG

class DeepfakeDetectorFusion(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, fusion="concat"):
        super().__init__()
        self.fusion = fusion

        # branch LBP
        self.xception_lbp = timm.create_model(CONFIG['model_name'], pretrained=True)
        num_features = self.xception_lbp.get_classifier().in_features
        self.xception_lbp.reset_classifier(0)

        # branch STA
        self.sta = SpatiotemporalAttention(in_channels=3)
        self.xception_sta = timm.create_model(CONFIG['model_name'], pretrained=True)
        self.xception_sta.reset_classifier(0)

        # fusion
        if fusion == "concat":
            fused_dim = num_features * 2
        elif fusion == "add":
            fused_dim = num_features
        else:
            raise ValueError("fusion must be 'concat' or 'add'")

        # temporal modeling
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.5
        )

        # predictor
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x_lbp, x_raw):
        """
        x_lbp : [B, T, C, H, W] (sudah diproses LBP)
        x_raw : [B, T, C, H, W] (raw frames untuk STA)
        """
        b, t, c, h, w = x_lbp.shape

        # brach lbp
        feat_lbp = self.xception_lbp(x_lbp.view(b*t, c, h, w))  # [B*T, F]
        feat_lbp = feat_lbp.view(b, t, -1)

        # branch sta
        x_sta = self.sta(x_raw)  # [B, T, C, H, W]
        feat_sta = self.xception_sta(x_sta.view(b*t, c, h, w))  # [B*T, F]
        feat_sta = feat_sta.view(b, t, -1)

        # fusion
        if self.fusion == "concat":
            fused = torch.cat([feat_lbp, feat_sta], dim=-1)  # [B, T, 2F]
        else:  # add
            fused = feat_lbp + feat_sta

        # temporal modeling
        lstm_out, _ = self.lstm(fused)
        last_output = lstm_out[:, -1, :]

        # classifier
        out = self.fc(last_output)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)