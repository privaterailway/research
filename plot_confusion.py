#!/usr/bin/env python3
"""Re-draw the confusion matrices from <OUTPUT_DIR>/scenario_*/predictions_test.npz.

(The training scripts already produce these; use this to regenerate them.)
"""
import argparse
import os
from os.path import join

import numpy as np

import reporting
from config import get_configuration

if __name__ == '__main__':
    cfg = get_configuration()
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=cfg['OUTPUT_DIR'])
    args = ap.parse_args()

    for s in ('a', 'b'):
        d = join(args.output_dir, f'scenario_{s}')
        path = join(d, 'predictions_test.npz')
        if not os.path.exists(path):
            print(f"Skipping scenario {s}: {path} not found")
            continue
        z = np.load(path, allow_pickle=False)
        classes = [str(c) for c in z['classes']]
        acc = 100.0 * float((z['y_true'] == z['y_pred']).mean())
        reporting.plot_confusion(z['y_true'], z['y_pred'], classes, acc, s, d, 'test')
        print(f"Re-drew confusion matrices for scenario {s} in {d}")
