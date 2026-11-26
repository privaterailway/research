#!/usr/bin/env python3
import argparse
import os
from math import ceil

from loguru import logger as log
from tqdm import tqdm
from encoder import ENCODER_FACTORY
from os import mkdir, listdir
from os.path import join, isdir

DATASET_DIR = '_dataset'
QUANTITY = 1024  # Количество шифротекста на каждый класс
TEXT_SIZE = 1024  # Размер текста в байтах
KEY_REUSE_EVERY = 8  # One key is shared by this many consecutive samples of a class (key group = `key_id`)
KEY_SIZE = 32  # bytes; every encoder truncates to the size it needs
IV_SIZE = 16  # bytes; every encoder truncates to the size it needs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the ciphertext dataset.")
    parser.add_argument(
        '--ciphers', nargs='+', metavar='NAME', default=None,
        help="Generate only these ciphers (default: all). Available: {}".format(', '.join(ENCODER_FACTORY)))
    parser.add_argument(
        '--missing-only', action='store_true',
        help="Skip ciphers whose class folder already contains a full set of samples "
             "(useful after adding new algorithms, keeps existing data untouched).")
    return parser.parse_args()


def is_class_complete(dir_path):
    return isdir(dir_path) and len([f for f in listdir(dir_path) if f.endswith('.bin')]) >= QUANTITY


def main():
    args = parse_args()

    selected = list(ENCODER_FACTORY.keys()) if args.ciphers is None else args.ciphers
    unknown = [c for c in selected if c not in ENCODER_FACTORY]
    if unknown:
        raise SystemExit("Unknown cipher(s): {}. Available: {}".format(', '.join(unknown), ', '.join(ENCODER_FACTORY)))

    log.info("Data generation started.")

    if not isdir(DATASET_DIR):
        mkdir(DATASET_DIR)

    # One IV for the whole dataset (same as the original setup)
    IV = os.urandom(IV_SIZE)

    for enc_name in selected:
        enc_factory = ENCODER_FACTORY[enc_name]
        dir_path = join(DATASET_DIR, enc_name)

        if args.missing_only and is_class_complete(dir_path):
            log.info(f"Skipping {enc_name}: dataset already present.")
            continue

        log.info(f"Generating data for {enc_name}...")

        if not isdir(dir_path):
            mkdir(dir_path)

        encoder = enc_factory()

        # Key material is generated INDEPENDENTLY for every cipher (nothing is shared between
        # classes) and a key is only shared inside one key group of this class.
        keys = [os.urandom(KEY_SIZE) for _ in range(ceil(QUANTITY / KEY_REUSE_EVERY))]

        for i in tqdm(range(QUANTITY), desc=f"[{enc_name}] Generating", unit="item"):
            key_id = i // KEY_REUSE_EVERY

            # A fresh random plaintext for every sample and every cipher: samples are neither
            # shared between classes nor structured (the old counter plaintext was ~all zeros).
            text = os.urandom(TEXT_SIZE)

            encoder.generate(keys[key_id], IV)
            ciphertext = encoder.encrypt(text)

            # Save the ciphertext (only the key group id is stored in the file name)
            with open(join(dir_path, f"{i + 1}_key{key_id}.bin"), "wb") as f:
                f.write(ciphertext)

    log.info("Data generation completed.")


if __name__ == '__main__':
    main()
