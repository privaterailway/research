"""Train / validation / test splits for both scenarios.

Scenario A (key reuse):   stratified random split over samples. The same keys
                          appear in train, val and test.
Scenario B (key disjoint): split over key ids. No key is shared between
                          train, val and test, so the test set measures
                          generalisation to unseen keys.

The splits are deterministic for a given seed and are shared by training,
evaluation and plotting (nothing is re-derived anywhere else).
"""
import json
from collections import defaultdict

import numpy as np


def _sizes(n, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def random_stratified_split(dataset, train_ratio, val_ratio, test_ratio, seed):
    """Scenario A: per-class random split (every class keeps the same proportions)."""
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for idx in range(len(dataset)):
        by_class[dataset.samples[idx][1]].append(idx)

    train, val, test = [], [], []
    for label in sorted(by_class):
        idxs = np.array(by_class[label])
        rng.shuffle(idxs)
        n_train, n_val, _ = _sizes(len(idxs), train_ratio, val_ratio, test_ratio)
        train.extend(idxs[:n_train].tolist())
        val.extend(idxs[n_train:n_train + n_val].tolist())
        test.extend(idxs[n_train + n_val:].tolist())
    return sorted(train), sorted(val), sorted(test)


def key_disjoint_split(dataset, train_ratio, val_ratio, test_ratio, seed):
    """Scenario B: split by key id, so no key is ever shared between the subsets."""
    by_key = defaultdict(list)
    for idx in range(len(dataset)):
        by_key[dataset.get_key_id(idx)].append(idx)

    keys = sorted(by_key)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n_train, n_val, _ = _sizes(len(keys), train_ratio, val_ratio, test_ratio)

    key_sets = {
        'train': keys[:n_train],
        'val': keys[n_train:n_train + n_val],
        'test': keys[n_train + n_val:],
    }
    out = {name: sorted(i for k in ks for i in by_key[k]) for name, ks in key_sets.items()}

    # Hard guarantee, checked every run
    assert not (set(key_sets['train']) & set(key_sets['val']))
    assert not (set(key_sets['train']) & set(key_sets['test']))
    assert not (set(key_sets['val']) & set(key_sets['test']))
    return out['train'], out['val'], out['test']


def make_splits(dataset, scenario, train_ratio, val_ratio, test_ratio, seed):
    fn = {'a': random_stratified_split, 'b': key_disjoint_split}[scenario]
    train, val, test = fn(dataset, train_ratio, val_ratio, test_ratio, seed)
    # No sample may appear in two subsets
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(dataset)
    return train, val, test


def split_summary(dataset, train, val, test):
    """Class counts + key counts per subset (goes into the paper's dataset table)."""
    rows = {}
    for name, idxs in (('train', train), ('val', val), ('test', test)):
        labels = [dataset.samples[i][1] for i in idxs]
        keys = {dataset.get_key_id(i) for i in idxs}
        counts = {dataset.classes[c]: labels.count(c) for c in range(len(dataset.classes))}
        rows[name] = {'samples': len(idxs), 'unique_keys': len(keys), 'per_class': counts}
    shared = {
        'train_val': len({dataset.get_key_id(i) for i in train} & {dataset.get_key_id(i) for i in val}),
        'train_test': len({dataset.get_key_id(i) for i in train} & {dataset.get_key_id(i) for i in test}),
        'val_test': len({dataset.get_key_id(i) for i in val} & {dataset.get_key_id(i) for i in test}),
    }
    rows['shared_keys_between'] = shared
    return rows


def save_split_manifest(dataset, train, val, test, path):
    manifest = {
        name: [dataset.samples[i][0].split('_dataset')[-1].lstrip('/\\') for i in idxs]
        for name, idxs in (('train', train), ('val', val), ('test', test))
    }
    with open(path, 'w') as f:
        json.dump(manifest, f)
