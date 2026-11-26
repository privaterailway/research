#!/usr/bin/env python3
"""Re-draw the accuracy / loss curves from <OUTPUT_DIR>/scenario_*/history.csv.

(The training scripts already produce these; use this to regenerate them,
e.g. after changing the plot style.)
"""
import argparse
import os
from os.path import join

import pandas as pd

import reporting
from config import get_configuration
from encoder import ENCODER_FACTORY

if __name__ == '__main__':
    cfg = get_configuration()
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=cfg['OUTPUT_DIR'])
    args = ap.parse_args()

    baseline = 100.0 / len(ENCODER_FACTORY)
    for s in ('a', 'b'):
        d = join(args.output_dir, f'scenario_{s}')
        path = join(d, 'history.csv')
        if not os.path.exists(path):
            print(f"Skipping scenario {s}: {path} not found")
            continue
        reporting.plot_training_curves(pd.read_csv(path), s, d, baseline)
        print(f"Re-drew curves for scenario {s} in {d}")
