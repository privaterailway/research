#!/usr/bin/env python3
import argparse
import os

from loguru import logger as log
from tqdm import tqdm
from encoder import ENCODER_FACTORY
from os import mkdir, listdir
from os.path import join, isdir

DATASET_DIR = '_dataset'
QUANTITY = 1024  # Количество шифротекста на каждый класс
TEXT_SIZE = 1024  # Размер текста в байтах
UPDATE_IV_AND_KEY_EVERY = 8  # Каждые n раз обновляем ключ и IV


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

    # Генерация текстов
    TEXTS = list([i.to_bytes(TEXT_SIZE, byteorder="big") for i in range(QUANTITY)])

    # Генерация ключа
    KEYS = list([os.urandom(32) for _ in range(QUANTITY // UPDATE_IV_AND_KEY_EVERY)])
    IV = os.urandom(16)

    for enc_name in selected:
        enc_factory = ENCODER_FACTORY[enc_name]
        dir_path = join(DATASET_DIR, enc_name)

        if args.missing_only and is_class_complete(dir_path):
            log.info(f"Skipping {enc_name}: dataset already present.")
            continue

        log.info(f"Generating data for {enc_name}...")

        if not isdir(dir_path):
            mkdir(dir_path)

        # Generate texts
        encoder = enc_factory()

        for i in tqdm(range(QUANTITY), desc=f"[{enc_name}] Generating", unit="item"):
            text = TEXTS[i]

            key_id = i // UPDATE_IV_AND_KEY_EVERY

            encoder.generate(KEYS[key_id], IV) # Генерируем ключ и IV на основе одних их тех же данных
            ciphertext = encoder.encrypt(text)

            # Сохраняем зашифрованные данные в файл
            with open(join(dir_path, f"{i + 1}_key{key_id}.bin"), "wb") as f:
                f.write(ciphertext)

    log.info("Data generation completed.")


if __name__ == '__main__':
    main()
