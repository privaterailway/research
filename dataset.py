import hashlib
import re
from os import listdir, stat
from os.path import exists, join

import numpy as np
import torch
from loguru import logger as log
from torch.utils.data import Dataset as TorchDataset

from encoder import ENCODER_FACTORY
from features import NUM_FEATURES, extract_features


class Dataset(TorchDataset):
    """Ciphertext dataset. Every .bin file is one sample, the folder name is the class.

    All files are read once and cached in memory (the whole dataset is only a few MB).
    """

    def __init__(self, root="_dataset", use_features=True):
        self.samples = []  # (file_path, label, key_id)
        self.classes = sorted(ENCODER_FACTORY.keys())
        self.use_features = use_features
        self.root = root

        data, labels, key_ids, raws = [], [], [], []
        sig = hashlib.sha256()
        for cipher_name in self.classes:
            class_path = join(root, cipher_name)
            for filename in sorted(listdir(class_path)):
                if not filename.endswith(".bin"):
                    continue
                fp = join(class_path, filename)
                with open(fp, "rb") as f:
                    raw = f.read()

                match = re.search(r'_key(\d+)\.bin', filename)
                key_id = int(match.group(1)) if match else -1
                label = self.classes.index(cipher_name)

                st = stat(fp)
                sig.update(f"{fp}|{st.st_size}|{st.st_mtime_ns}".encode())

                self.samples.append((fp, label, key_id))
                if use_features:
                    raws.append(raw)
                data.append(torch.tensor(list(raw), dtype=torch.long))
                labels.append(label)
                key_ids.append(key_id)

        self.data = torch.stack(data)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.key_ids = torch.tensor(key_ids, dtype=torch.long)

        # Hand-crafted features G1-G5 (raw, un-normalised). They are normalised later with
        # statistics of the TRAIN split only (see set_feature_norm) to avoid leakage.
        self.features = None
        if use_features:
            self.features = torch.from_numpy(self._load_or_compute_features(raws, sig.hexdigest()))
        self._feat_mean = None
        self._feat_std = None

    def _load_or_compute_features(self, raws, signature):
        cache = join(self.root, ".features_cache.npz")
        if exists(cache):
            try:
                z = np.load(cache)
                if str(z["signature"]) == signature and z["features"].shape == (len(raws), NUM_FEATURES):
                    log.info(f"Loaded cached features from {cache}")
                    return z["features"].astype(np.float32)
            except Exception:
                pass
        log.info(f"Computing {NUM_FEATURES} features (G1-G5) for {len(raws)} samples ...")
        feats = np.stack([extract_features(r) for r in raws]).astype(np.float32)
        try:
            np.savez_compressed(cache, features=feats, signature=np.array(signature))
        except OSError:
            pass
        return feats

    def set_feature_norm(self, train_indices):
        """Z-score the features using mean/std of the training samples only."""
        if self.features is None:
            return
        tr = self.features[list(train_indices)]
        self._feat_mean = tr.mean(dim=0)
        self._feat_std = tr.std(dim=0).clamp_min(1e-6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.features is None:
            feat = torch.zeros(0)
        else:
            feat = self.features[idx]
            if self._feat_mean is not None:
                feat = (feat - self._feat_mean) / self._feat_std
        return self.data[idx], feat, int(self.labels[idx])

    def get_key_id(self, idx):
        """Return key_id for sample at index idx"""
        return self.samples[idx][2]
