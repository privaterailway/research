import torch
from torch import nn


class CipherClassifier(nn.Module):
    def __init__(self, num_classes=4, d_model=32, hidden=96, num_layers=1, dropout=0.1, num_features=0):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)

        self.conv1d_8 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=8, stride=2)
        self.conv1d_16 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=16, stride=2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * d_model, nhead=8, dim_feedforward=hidden, dropout=dropout,
            activation='relu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Branch for the hand-crafted G1-G5 features (optional)
        self.num_features = num_features
        feat_out = 0
        if num_features > 0:
            feat_out = 64
            self.feature_net = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, feat_out),
                nn.ReLU(),
            )

        self.output = nn.Sequential(
            nn.Linear(2 * d_model + feat_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, feats=None):
        x = self.embed(x)                 # [B, L, d_model]
        x = x.transpose(1, 2)             # [B, d_model, L]

        x8  = self.conv1d_8(x)            # [B, d_model, L8]
        x16 = self.conv1d_16(x)           # [B, d_model, L16]
        minL = min(x8.size(2), x16.size(2))
        x = torch.cat([x8[:, :, :minL], x16[:, :, :minL]], dim=1)  # [B, 2*d_model, L']

        x = x.transpose(1, 2)             # [B, L', 2*d_model]
        x = self.encoder(x)               # [B, L', 2*d_model]
        x = x.mean(dim=1)                 # [B, 2*d_model]

        if self.num_features > 0:
            x = torch.cat([x, self.feature_net(feats)], dim=1)

        return self.output(x)
