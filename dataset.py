import hashlib
import re
from os import listdir
from os.path import exists, join

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

import features as feat
from encoder import ENCODER_FACTORY


class Dataset(TorchDataset):
    """Ciphertext dataset. Every .bin file is one sample, the folder name is the class.

    All files are read once and cached in memory (the whole dataset is only a few MB).

    Each item is ``(raw_bytes [L], statistical_features [F], label)``. The statistical
    features (groups G1-G5, see ``features.py``) are computed once and cached next to the
    data (``<root>/.features_<groups>.npz``). If ``use_features=False`` then F = 0.

    The features are stored *raw*. Call ``standardize_features(train_idx)`` after the split
    is known: it z-scores them with statistics of the training subset only (no leakage).
    """

    def __init__(self, root="_dataset", use_features=True, feature_groups=None, cache_features=True):
        self.samples = []  # (file_path, label, key_id)
        self.classes = sorted(ENCODER_FACTORY.keys())

        data, labels, key_ids, raws = [], [], [], []
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

                self.samples.append((fp, label, key_id))
                raws.append(raw)
                data.append(torch.tensor(list(raw), dtype=torch.long))
                labels.append(label)
                key_ids.append(key_id)

        self.data = torch.stack(data)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.key_ids = torch.tensor(key_ids, dtype=torch.long)

        # ---- statistical features
        self.use_features = bool(use_features)
        self.feature_groups = tuple(feature_groups) if feature_groups else feat.GROUPS
        if self.use_features:
            self.feature_names = feat.get_feature_names(self.feature_groups)
            self.features = torch.from_numpy(self._load_or_compute_features(root, raws, cache_features))
        else:
            self.feature_names = []
            self.features = torch.zeros((len(self.samples), 0), dtype=torch.float32)

    # ------------------------------------------------------------------ features
    def _load_or_compute_features(self, root, raws, use_cache):
        digest = hashlib.sha1()
        for raw in raws:
            digest.update(raw)
        digest = f"v{feat.FEATURE_VERSION}:{digest.hexdigest()}"
        cache_path = join(root, f".features_{'-'.join(self.feature_groups)}.npz")

        if use_cache and exists(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                if str(z['digest']) == digest and z['features'].shape == (len(raws), len(self.feature_names)):
                    return z['features'].astype(np.float32)
            except Exception:
                pass  # corrupt / stale cache -> recompute

        rows = [feat.extract_features(raw, self.feature_groups)
                for raw in tqdm(raws, desc="Extracting statistical features", unit="sample")]
        matrix = np.stack(rows).astype(np.float32)

        if use_cache:
            try:
                np.savez_compressed(cache_path, features=matrix, digest=np.array(digest))
            except OSError:
                pass  # read-only dataset folder: just skip caching
        return matrix

    def standardize_features(self, train_idx):
        """z-score the features with mean / std of the training subset; clip to [-10, 10].

        Returns (mean, std) as numpy arrays (or None if there are no features).
        """
        if self.features.shape[1] == 0:
            return None
        train = self.features[torch.as_tensor(list(train_idx), dtype=torch.long)]
        mean = train.mean(dim=0)
        std = train.std(dim=0)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)  # constant feature -> 0 after centering
        self.features = ((self.features - mean) / std).clamp(-10.0, 10.0)
        return mean.numpy(), std.numpy()

    # ------------------------------------------------------------------ torch API
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data[idx], self.features[idx], int(self.labels[idx])

    def get_key_id(self, idx):
        """Return key_id for sample at index idx"""
        return self.samples[idx][2]
