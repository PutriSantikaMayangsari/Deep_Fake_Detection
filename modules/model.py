# modules/model.py

import torch
import torch.nn as nn
import timm
from .attention import SpatiotemporalAttention
from config import CONFIG # Import CONFIG untuk nama model

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512):
        super(DeepfakeDetector, self).__init__()
        self.attention = SpatiotemporalAttention(in_channels=3)
        
        self.xception = timm.create_model(CONFIG['model_name'], pretrained=True)
        num_features = self.xception.get_classifier().in_features
        self.xception.reset_classifier(0) # Hapus classifier asli
        
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = self.attention(x)
        x = x.view(b * t, c, h, w)
        features = self.xception(x)
        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out