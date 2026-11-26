"""Metrics, tables and figures for the paper.

Everything is written to the outputs folder as PNG (quick look) and PDF (vector,
for LaTeX), plus CSV / JSON / .tex tables.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

SCENARIO_TITLES = {
    'a': 'Scenario A: Key Reuse (stratified random split)',
    'b': 'Scenario B: Key-Disjoint Split',
}
SCENARIO_SHORT = {'a': 'A: Key reuse', 'b': 'B: Key-disjoint'}


# ----------------------------------------------------------------------------- io helpers
def save_fig(fig, base_path):
    """Save the figure as <base>.png (300 dpi) and <base>.pdf."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------------- metrics
def wilson_interval(correct, n, z=1.96):
    """95% Wilson score interval for an accuracy (returned in percent)."""
    if n == 0:
        return float('nan'), float('nan')
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def compute_metrics(y_true, y_pred, probs, class_names):
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    labels = list(range(len(class_names)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    ci_low, ci_high = wilson_interval(correct, n)

    metrics = {
        'n_samples': n,
        'accuracy': 100.0 * correct / n,
        'accuracy_ci95_low': ci_low,
        'accuracy_ci95_high': ci_high,
        'balanced_accuracy': 100.0 * balanced_accuracy_score(y_true, y_pred),
        'macro_precision': 100.0 * float(np.mean(prec)),
        'macro_recall': 100.0 * float(np.mean(rec)),
        'macro_f1': 100.0 * float(np.mean(f1)),
        'weighted_f1': 100.0 * float(f1_w),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'random_baseline': 100.0 / len(class_names),
    }
    if probs is not None:
        top3 = np.argsort(-probs, axis=1)[:, :3]
        metrics['top3_accuracy'] = 100.0 * float(np.mean([t in row for t, row in zip(y_true, top3)]))
        try:
            metrics['macro_roc_auc_ovr'] = float(roc_auc_score(
                y_true, probs, multi_class='ovr', average='macro', labels=labels))
        except ValueError:  # a class missing from y_true
            metrics['macro_roc_auc_ovr'] = float('nan')

    report = pd.DataFrame({
        'class': class_names,
        'precision': 100 * prec, 'recall': 100 * rec, 'f1': 100 * f1, 'support': support,
    })
    return metrics, report


# ----------------------------------------------------------------------------- figures
def plot_training_curves(history, scenario, out_dir, baseline):
    """history: DataFrame with epoch, train_loss, val_loss, train_acc, val_acc."""
    best_epoch = int(history.loc[history['val_loss'].idxmin(), 'epoch'])

    # accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['train_acc'], lw=2, color='#1f77b4', label='Training')
    ax.plot(history['epoch'], history['val_acc'], lw=2, color='#ff7f0e', label='Validation')
    ax.axhline(baseline, color='red', ls='--', lw=1.3, alpha=0.7, label=f'Random baseline ({baseline:.2f}%)')
    ax.axvline(best_epoch, color='gray', ls=':', lw=1.3, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(SCENARIO_TITLES[scenario], pad=12)
    ax.set_xlim(1, history['epoch'].max())
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'accuracy_curve_scenario_{scenario}'))

    # loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['train_loss'], lw=2, color='#1f77b4', label='Training')
    ax.plot(history['epoch'], history['val_loss'], lw=2, color='#ff7f0e', label='Validation')
    ax.axvline(best_epoch, color='gray', ls=':', lw=1.3, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title(SCENARIO_TITLES[scenario] + ' - Loss', pad=12)
    ax.set_xlim(1, history['epoch'].max())
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'loss_curve_scenario_{scenario}'))


def plot_confusion(y_true, y_pred, class_names, accuracy, scenario, out_dir, split_name='test'):
    """Counts and row-normalised confusion matrices (PNG + PDF + CSV)."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, f'confusion_matrix_{split_name}_scenario_{scenario}.csv'))

    size = max(8, 0.65 * n + 3)
    for kind, mat, fmt, label in (('counts', cm, 'd', 'Count'), ('normalized', cm_norm, '.2f', 'Fraction of true class')):
        fig, ax = plt.subplots(figsize=(size, size * 0.92))
        sns.heatmap(mat, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names, cbar_kws={'label': label}, square=True,
                    linewidths=0.4, linecolor='lightgray', ax=ax,
                    annot_kws={'size': 8 if n > 8 else 10},
                    vmin=0, vmax=1 if kind == 'normalized' else None)
        ax.set_title(f'{SCENARIO_TITLES[scenario]}\n{split_name.capitalize()} accuracy: {accuracy:.2f}%',
                     pad=14, weight='bold')
        ax.set_ylabel('True label', weight='bold')
        ax.set_xlabel('Predicted label', weight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'confusion_{kind}_{split_name}_scenario_{scenario}'))
    return cm


def plot_per_class_f1(report, scenario, out_dir, split_name='test'):
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(report) + 3), 4.8))
    ax.bar(report['class'], report['f1'], color='#1f77b4', edgecolor='black', lw=0.5)
    ax.axhline(100.0 / len(report), color='red', ls='--', lw=1.3, alpha=0.7,
               label=f'Random baseline ({100.0 / len(report):.2f}%)')
    ax.set_ylabel('F1 score (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'{SCENARIO_TITLES[scenario]}\nPer-class F1 ({split_name} set)', pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'per_class_f1_{split_name}_scenario_{scenario}'))


def plot_scenario_comparison(results, out_dir):
    """results: {'a': {...test metrics...}, 'b': {...}} -> bar chart with 95% CI + baseline."""
    keys = [k for k in ('a', 'b') if k in results]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    accs = [results[k]['test']['accuracy'] for k in keys]
    lo = [results[k]['test']['accuracy'] - results[k]['test']['accuracy_ci95_low'] for k in keys]
    hi = [results[k]['test']['accuracy_ci95_high'] - results[k]['test']['accuracy'] for k in keys]
    bars = ax.bar([SCENARIO_SHORT[k] for k in keys], accs, yerr=[lo, hi], capsize=6,
                  color=['#2ca02c', '#d62728'][:len(keys)], edgecolor='black', lw=0.6)
    baseline = results[keys[0]]['test']['random_baseline']
    ax.axhline(baseline, color='black', ls='--', lw=1.3, label=f'Random baseline ({baseline:.2f}%)')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 2.5, f'{acc:.2f}%', ha='center', weight='bold')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Test accuracy by scenario (95% Wilson CI)', pad=12)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'scenario_comparison_test_accuracy'))


# ----------------------------------------------------------------------------- tables
def summary_table(results):
    rows = []
    for k in ('a', 'b'):
        if k not in results:
            continue
        r = results[k]
        rows.append({
            'Scenario': SCENARIO_SHORT[k],
            'Train acc (%)': r['train']['accuracy'],
            'Val acc (%)': r['val']['accuracy'],
            'Test acc (%)': r['test']['accuracy'],
            'Test 95% CI': f"[{r['test']['accuracy_ci95_low']:.2f}, {r['test']['accuracy_ci95_high']:.2f}]",
            'Balanced acc (%)': r['test']['balanced_accuracy'],
            'Macro-F1 (%)': r['test']['macro_f1'],
            'MCC': r['test']['mcc'],
            'Top-3 acc (%)': r['test'].get('top3_accuracy', float('nan')),
            'Macro AUC': r['test'].get('macro_roc_auc_ovr', float('nan')),
            'Random baseline (%)': r['test']['random_baseline'],
            'Best epoch': r['best_epoch'],
        })
    return pd.DataFrame(rows)


def write_latex(df, path, caption, label, float_format='%.2f'):
    latex = df.to_latex(index=False, float_format=lambda x: float_format % x,
                        caption=caption, label=label, escape=True, column_format='l' + 'r' * (df.shape[1] - 1))
    with open(path, 'w') as f:
        f.write(latex)
