#!/usr/bin/env python3
"""Combine scenario A and B results into the paper summary table + comparison figure.

Writes to <OUTPUT_DIR>/summary/ :
    results_summary.csv / .tex / .json, scenario_comparison_test_accuracy.(png|pdf)
"""
import argparse
import json
import os
from os.path import join

import reporting
from config import get_configuration


def main():
    cfg = get_configuration()
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=cfg['OUTPUT_DIR'])
    args = ap.parse_args()

    results = {}
    for s in ('a', 'b'):
        path = join(args.output_dir, f'scenario_{s}', 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                results[s] = json.load(f)
        else:
            print(f"Warning: {path} not found - run train_scenario_{s}.py first")
    if not results:
        raise SystemExit("No results found.")

    out_dir = join(args.output_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)

    df = reporting.summary_table(results)
    df.to_csv(join(out_dir, 'results_summary.csv'), index=False)
    reporting.write_latex(df[['Scenario', 'Train acc (%)', 'Val acc (%)', 'Test acc (%)', 'Test 95% CI',
                              'Macro-F1 (%)', 'MCC']],
                          join(out_dir, 'results_summary.tex'),
                          caption='Classification results under key reuse (A) and key-disjoint (B) splits.',
                          label='tab:results_summary')
    reporting.write_json(results, join(out_dir, 'results_summary.json'))
    reporting.plot_scenario_comparison(results, out_dir)

    print(df.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
