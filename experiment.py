"""Shared train / validate / test pipeline for both scenarios.

Writes everything the paper needs to  <OUTPUT_DIR>/scenario_<a|b>/  :
    config.json, split_summary.json, split_manifest.json, history.csv, results.json,
    classification_report_{val,test}.csv/.tex, confusion_matrix_test_*.csv,
    predictions_{val,test}.npz, model_best.pth, train.log and all figures (PNG + PDF).
"""
import json
import os
import random
import time
from os.path import abspath, join

import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from torch.utils.data import DataLoader, Subset

import reporting
import splits
from config import get_configuration
from dataset import Dataset
from features import NUM_FEATURES
from model import CipherClassifier


def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, F, labels in loader:
            X = X.to(device, non_blocking=True)
            F = F.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            out = model(X, F)
            loss = loss_fn(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            total += len(labels)
            correct += (out.argmax(dim=1) == labels).sum().item()
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


@torch.no_grad()
def predict(model, loader, device):
    """Returns y_true, y_pred, softmax probabilities."""
    model.eval()
    ys, probs = [], []
    for X, F, labels in loader:
        out = model(X.to(device, non_blocking=True), F.to(device, non_blocking=True))
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
        ys.append(labels.numpy())
    probs = np.concatenate(probs)
    return np.concatenate(ys), probs.argmax(axis=1), probs


def run_scenario(scenario, seed=None, epochs=None, output_dir=None, dataset_path=None, batch_size=None):
    assert scenario in ('a', 'b')
    cfg = get_configuration()
    seed = cfg['SEED'] if seed is None else seed
    epochs = int(cfg['EPOCHS'] if epochs is None else epochs)
    batch_size = int(cfg['BATCH_SIZE'] if batch_size is None else batch_size)
    out_root = output_dir or cfg['OUTPUT_DIR']
    out_dir = abspath(join(out_root, f'scenario_{scenario}'))
    os.makedirs(out_dir, exist_ok=True)

    log_sink = log.add(join(out_dir, 'train.log'), format="{time:HH:mm:ss} | {level} | {message}")
    set_seed(seed)
    device = get_current_device()

    # model hyper-parameters (unchanged from the original scripts)
    d_model, hidden, num_layers, dropout = 16, 32, 2, 0.2

    log.info("=" * 60)
    log.info(f"{reporting.SCENARIO_TITLES[scenario]}  (seed={seed}, device={device})")
    log.info("=" * 60)

    # ------------------------------------------------------------------ data + split
    dataset_path = abspath(dataset_path or cfg.get('DATASET_PATH', '_dataset'))
    log.info(f"Loading dataset from: {dataset_path}")
    use_features = bool(cfg.get('USE_FEATURES', True))
    full = Dataset(root=dataset_path, use_features=use_features)
    class_names = full.classes
    num_classes = len(class_names)

    train_idx, val_idx, test_idx = splits.make_splits(
        full, scenario, cfg['TRAIN_RATIO'], cfg['VAL_RATIO'], cfg['TEST_RATIO'], seed)
    # Feature normalisation statistics come from the TRAIN split only (no val/test leakage)
    full.set_feature_norm(train_idx)
    summary = splits.split_summary(full, train_idx, val_idx, test_idx)
    reporting.write_json(summary, join(out_dir, 'split_summary.json'))
    splits.save_split_manifest(full, train_idx, val_idx, test_idx, join(out_dir, 'split_manifest.json'))
    for name in ('train', 'val', 'test'):
        log.info(f"{name:>5}: {summary[name]['samples']} samples, {summary[name]['unique_keys']} keys")
    log.info(f"Keys shared between subsets: {summary['shared_keys_between']}")
    if scenario == 'b':
        assert all(v == 0 for v in summary['shared_keys_between'].values()), "key leakage between splits!"

    pin = device.type == 'cuda'
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, generator=gen)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(Subset(full, test_idx), batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)
    train_eval_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=pin)

    # ------------------------------------------------------------------ model
    model = CipherClassifier(num_classes=num_classes, d_model=d_model, hidden=hidden,
                             num_layers=num_layers, dropout=dropout,
                             num_features=NUM_FEATURES if use_features else 0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['LEARNING_RATE']),
                                  weight_decay=float(cfg['WEIGHT_DECAY']))
    loss_fn = torch.nn.CrossEntropyLoss()

    run_config = {
        'scenario': scenario, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
        'learning_rate': cfg['LEARNING_RATE'], 'weight_decay': cfg['WEIGHT_DECAY'],
        'd_model': d_model, 'hidden': hidden, 'num_layers': num_layers, 'dropout': dropout,
        'optimizer': 'AdamW', 'model_selection': 'lowest validation loss',
        'split_ratios': [cfg['TRAIN_RATIO'], cfg['VAL_RATIO'], cfg['TEST_RATIO']],
        'use_features': use_features, 'num_features': NUM_FEATURES if use_features else 0,
        'num_classes': num_classes, 'classes': class_names, 'trainable_parameters': n_params,
        'device': str(device), 'torch_version': torch.__version__,
    }
    reporting.write_json(run_config, join(out_dir, 'config.json'))
    log.info(f"Model parameters: {n_params:,}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=join(out_dir, 'tensorboard'))
    except Exception:
        pass  # tensorboard is optional now; history.csv is the source of truth

    # ------------------------------------------------------------------ training
    ckpt_path = join(out_dir, 'model_best.pth')
    best_val_loss, best_epoch = float('inf'), 0
    rows = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, loss_fn)
        rows.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                     'val_loss': va_loss, 'val_acc': va_acc})
        marker = ''
        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), ckpt_path)
            marker = '  <- best'
        log.info(f"Epoch {epoch:>3}/{epochs} | train {tr_loss:.4f} / {tr_acc:6.2f}% | "
                 f"val {va_loss:.4f} / {va_acc:6.2f}%{marker}")
        if writer:
            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Loss/val", va_loss, epoch)
            writer.add_scalar("Accuracy/train %", tr_acc, epoch)
            writer.add_scalar("Accuracy/val %", va_acc, epoch)
    train_seconds = time.time() - t0
    if writer:
        writer.close()

    history = pd.DataFrame(rows)
    history.to_csv(join(out_dir, 'history.csv'), index=False)

    # ------------------------------------------------------------------ final evaluation (best-val-loss checkpoint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    results = {'scenario': scenario, 'seed': seed, 'best_epoch': best_epoch,
               'train_seconds': train_seconds, 'trainable_parameters': n_params}
    preds = {}
    for name, loader in (('train', train_eval_loader), ('val', val_loader), ('test', test_loader)):
        y, p, probs = predict(model, loader, device)
        metrics, report = reporting.compute_metrics(y, p, probs, class_names)
        results[name] = metrics
        preds[name] = (y, p, probs)
        if name != 'train':
            report.to_csv(join(out_dir, f'classification_report_{name}.csv'), index=False)
            reporting.write_latex(
                report, join(out_dir, f'classification_report_{name}.tex'),
                caption=f'Per-class results on the {name} set, {reporting.SCENARIO_TITLES[scenario]}.',
                label=f'tab:per_class_{name}_{scenario}')
            np.savez_compressed(join(out_dir, f'predictions_{name}.npz'),
                                y_true=y, y_pred=p, probs=probs, classes=np.array(class_names))
        log.info(f"{name:>5}: acc {metrics['accuracy']:.2f}% | macro-F1 {metrics['macro_f1']:.2f}% | "
                 f"MCC {metrics['mcc']:.3f}")
        if name == 'test':
            test_report = report

    results['hardest_classes_test'] = (test_report.sort_values('f1').head(3)['class'].tolist())
    reporting.write_json(results, join(out_dir, 'results.json'))

    # ------------------------------------------------------------------ figures
    baseline = 100.0 / num_classes
    reporting.plot_training_curves(history, scenario, out_dir, baseline)
    y, p, _ = preds['test']
    reporting.plot_confusion(y, p, class_names, results['test']['accuracy'], scenario, out_dir, 'test')
    reporting.plot_per_class_f1(test_report, scenario, out_dir, 'test')

    log.info(f"Best epoch: {best_epoch} | TEST accuracy: {results['test']['accuracy']:.2f}% "
             f"(random baseline {baseline:.2f}%)")
    log.info(f"All outputs saved to: {out_dir}")
    log.remove(log_sink)
    return results
