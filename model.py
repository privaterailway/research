import torch
from torch import nn


class CipherClassifier(nn.Module):
    """Hybrid ciphertext classifier.

    * byte branch    : embedding -> two Conv1d (k=8, k=16) -> Transformer encoder -> mean pool
    * feature branch : MLP over the standardized statistical features (G1-G5, see features.py)
    * the two representations are concatenated and fed to the classification head.

    Either branch can be switched off (``use_bytes=False`` / ``num_features=0``) for ablations.
    """

    def __init__(self, num_classes=4, d_model=32, hidden=96, num_layers=1, dropout=0.1,
                 num_features=0, use_bytes=True, feature_hidden=128, feature_dim=64):
        super().__init__()
        self.use_bytes = bool(use_bytes)
        self.num_features = int(num_features)
        if not self.use_bytes and self.num_features == 0:
            raise ValueError("At least one of the byte branch / statistical features must be enabled")

        head_in = 0

        if self.use_bytes:
            self.embed = nn.Embedding(256, d_model)

            self.conv1d_8 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=8, stride=2)
            self.conv1d_16 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=16, stride=2)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=2 * d_model, nhead=8, dim_feedforward=hidden, dropout=dropout,
                activation='relu', batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            head_in += 2 * d_model

        if self.num_features > 0:
            self.feature_net = nn.Sequential(
                nn.Linear(self.num_features, feature_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_hidden, feature_dim),
                nn.ReLU(),
            )
            head_in += feature_dim

        self.output = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, feats=None):
        parts = []

        if self.use_bytes:
            h = self.embed(x)                 # [B, L, d_model]
            h = h.transpose(1, 2)             # [B, d_model, L]

            h8 = self.conv1d_8(h)             # [B, d_model, L8]
            h16 = self.conv1d_16(h)           # [B, d_model, L16]
            minL = min(h8.size(2), h16.size(2))
            h = torch.cat([h8[:, :, :minL], h16[:, :, :minL]], dim=1)  # [B, 2*d_model, L']

            h = h.transpose(1, 2)             # [B, L', 2*d_model]
            h = self.encoder(h)               # [B, L', 2*d_model]
            parts.append(h.mean(dim=1))       # [B, 2*d_model]

        if self.num_features > 0:
            if feats is None:
                raise ValueError("Model was built with statistical features but forward() got feats=None")
            parts.append(self.feature_net(feats))  # [B, feature_dim]

        return self.output(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])
