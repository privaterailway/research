import torch


def get_configuration():
    return {
        'BATCH_SIZE': 128,
        'LEARNING_RATE': 0.0001,
        'WEIGHT_DECAY': 0.001,
        'EPOCHS': 50,
        'SEQUENCE_LENGTH': 8,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'BYTES_QUANTITY': 256,

        # ---- model inputs ----
        # Statistical features (see features.py):
        #   G1 byte distribution, G2 entropy/compressibility, G3 NIST SP 800-22 p-values,
        #   G4 serial dependence, G5 block structure.
        'USE_STAT_FEATURES': True,
        'FEATURE_GROUPS': ['G1', 'G2', 'G3', 'G4', 'G5'],   # drop entries for ablations
        'USE_BYTE_SEQUENCE': True,                          # False -> statistical features only

        # ---- split / output settings ----
        'SEED': 42,
        'TRAIN_RATIO': 0.70,
        'VAL_RATIO': 0.15,
        'TEST_RATIO': 0.15,
        'OUTPUT_DIR': 'outputs',
    }
