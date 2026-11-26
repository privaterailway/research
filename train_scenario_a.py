#!/usr/bin/env python3
"""Scenario A: key reuse. Stratified random train/val/test split (same keys in all subsets)."""
import argparse

from experiment import run_scenario

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--batch-size', type=int, default=None, help='lower this if you run out of memory')
    ap.add_argument('--dataset-path', default=None)
    args = ap.parse_args()
    run_scenario('a', seed=args.seed, epochs=args.epochs, output_dir=args.output_dir,
                 batch_size=args.batch_size, dataset_path=args.dataset_path)
