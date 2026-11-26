"""Hand-crafted statistical features of a ciphertext (groups G1 - G5).

    G1  Byte distribution        256-bin histogram, chi^2 vs. uniform, first four moments      (261)
    G2  Entropy / compressibility Shannon (Miller-Madow), Renyi-2, min-entropy, zlib, LZMA      (5)
    G3  Bit-level randomness     6 NIST SP 800-22 tests (cusum gives a forward and a backward
                                 p-value, so 7 numbers)                                         (7)
    G4  Serial dependence        byte autocorrelation, lags 1..32, + bigram chi^2 statistics   (35)
    G5  Block structure          b in {8, 16}: repeated / distinct aligned blocks, mean Hamming
                                 distance between consecutive blocks; + inferred block size    (7)

Everything is implemented with numpy / scipy / the standard library (no NIST
package needed) and works for ciphertexts of any length (the defaults are tuned
for ~1 KiB samples, i.e. 8192 bits).

Public API
----------
    extract_features(raw: bytes, groups=GROUPS) -> np.ndarray  (float32, 1-D)
    get_feature_names(groups=GROUPS)            -> list[str]   (same order as the vector)
"""
import lzma
import math
import zlib
from functools import lru_cache

import numpy as np
from scipy import special

FEATURE_VERSION = 1  # bump when anything below changes (invalidates cached features)

GROUPS = ('G1', 'G2', 'G3', 'G4', 'G5')
MAX_LAG = 32                      # G4: autocorrelation lags 1..MAX_LAG
BLOCK_SIZES = (8, 16)             # G5: block sizes in bytes
INFER_CANDIDATES = (4, 8, 16, 32)  # G5: candidates for the inferred block size
NIST_BLOCK_FREQ_M = 128           # G3: block length of the block-frequency test
NIST_APEN_M = 2                   # G3: pattern length of the approximate-entropy test
NEUTRAL_P = 0.5                   # p-value used when a NIST test is not applicable (sequence too short)

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


# ====================================================================== G1
def _g1(arr, n):
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    hist = counts / n

    expected = n / 256.0
    chi2 = float(((counts - expected) ** 2).sum() / expected)

    x = arr.astype(np.float64)
    mean = float(x.mean())
    var = float(x.var())
    std = math.sqrt(var)
    if std > 0:
        z = (x - mean) / std
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)  # excess kurtosis
    else:
        skew, kurt = 0.0, 0.0

    names = [f'g1_hist_{i:03d}' for i in range(256)]
    names += ['g1_chi2_uniform', 'g1_mean', 'g1_variance', 'g1_skewness', 'g1_kurtosis']
    return names, list(hist) + [chi2, mean, var, skew, kurt]


# ====================================================================== G2
def _g2(arr, raw, n):
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts[counts > 0] / n
    k_observed = p.size

    h_plugin = float(-(p * np.log2(p)).sum())
    # Miller-Madow: H_ML + (K - 1) / (2 n) nats  ->  divide by ln 2 for bits
    h_mm = h_plugin + (k_observed - 1) / (2.0 * n * math.log(2.0))
    renyi2 = float(-math.log2((p ** 2).sum()))
    h_min = float(-math.log2(p.max()))

    zlib_ratio = len(zlib.compress(raw, 9)) / n
    lzma_ratio = len(lzma.compress(
        raw, format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9}])) / n

    names = ['g2_shannon_miller_madow', 'g2_renyi2', 'g2_min_entropy', 'g2_zlib_ratio', 'g2_lzma_ratio']
    return names, [h_mm, renyi2, h_min, zlib_ratio, lzma_ratio]


# ====================================================================== G3  (NIST SP 800-22)
def _nist_frequency(bits):
    n = bits.size
    s = 2 * int(bits.sum()) - n
    return float(special.erfc(abs(s) / math.sqrt(n) / math.sqrt(2.0)))


def _nist_block_frequency(bits, m=NIST_BLOCK_FREQ_M):
    n_blocks = bits.size // m
    if n_blocks < 1:
        return float('nan')
    blocks = bits[:n_blocks * m].reshape(n_blocks, m)
    pi = blocks.mean(axis=1)
    chi2 = 4.0 * m * float(((pi - 0.5) ** 2).sum())
    return float(special.gammaincc(n_blocks / 2.0, chi2 / 2.0))


def _nist_runs(bits):
    n = bits.size
    pi = float(bits.sum()) / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):  # prerequisite frequency test failed -> p = 0 (NIST)
        return 0.0
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    return float(special.erfc(num / den))


def _longest_run_of_ones(row):
    padded = np.concatenate(([0], row, [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def _nist_longest_run(bits):
    n = bits.size
    if n < 128:
        return float('nan')
    if n < 6272:
        m, n_blocks, lo, hi = 8, 16, 1, 4
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        m, n_blocks, lo, hi = 128, 49, 4, 9
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        m, n_blocks, lo, hi = 10000, 75, 10, 16
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    pi = np.asarray(pi)

    blocks = bits[:n_blocks * m].reshape(n_blocks, m).astype(np.int8)
    longest = np.array([_longest_run_of_ones(r) for r in blocks])
    nu = np.bincount(np.clip(longest, lo, hi) - lo, minlength=hi - lo + 1).astype(np.float64)
    expected = n_blocks * pi
    chi2 = float(((nu - expected) ** 2 / expected).sum())
    return float(special.gammaincc((hi - lo) / 2.0, chi2 / 2.0))


def _nist_cusum(bits, reverse=False):
    n = bits.size
    x = 2 * bits.astype(np.int64) - 1
    if reverse:
        x = x[::-1]
    z = int(np.abs(np.cumsum(x)).max())
    if z == 0:
        return 0.0
    sqn = math.sqrt(n)
    k1 = np.arange(int(math.floor((-n / z + 1) / 4)), int(math.floor((n / z - 1) / 4)) + 1)
    k2 = np.arange(int(math.floor((-n / z - 3) / 4)), int(math.floor((n / z - 1) / 4)) + 1)
    s1 = (special.ndtr((4 * k1 + 1) * z / sqn) - special.ndtr((4 * k1 - 1) * z / sqn)).sum()
    s2 = (special.ndtr((4 * k2 + 3) * z / sqn) - special.ndtr((4 * k2 + 1) * z / sqn)).sum()
    return float(min(max(1.0 - s1 + s2, 0.0), 1.0))


def _apen_phi(bits, m):
    n = bits.size
    ext = np.concatenate([bits, bits[:m - 1]]).astype(np.int64)  # circular extension
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n]
    counts = np.bincount(idx, minlength=2 ** m).astype(np.float64)
    c = counts[counts > 0] / n
    return float((c * np.log(c)).sum())


def _nist_approximate_entropy(bits, m=NIST_APEN_M):
    n = bits.size
    apen = _apen_phi(bits, m) - _apen_phi(bits, m + 1)
    chi2 = 2.0 * n * (math.log(2.0) - apen)
    return float(special.gammaincc(2 ** (m - 1), chi2 / 2.0))


def _g3(arr):
    bits = np.unpackbits(arr).astype(np.int8)
    tests = [
        ('g3_nist_frequency', _nist_frequency(bits)),
        ('g3_nist_block_frequency', _nist_block_frequency(bits)),
        ('g3_nist_runs', _nist_runs(bits)),
        ('g3_nist_longest_run', _nist_longest_run(bits)),
        ('g3_nist_cusum_forward', _nist_cusum(bits, reverse=False)),
        ('g3_nist_cusum_backward', _nist_cusum(bits, reverse=True)),
        ('g3_nist_approx_entropy', _nist_approximate_entropy(bits)),
    ]
    names = [t[0] for t in tests]
    values = [v if np.isfinite(v) else NEUTRAL_P for _, v in tests]
    return names, values


# ====================================================================== G4
def _g4(arr, n):
    # --- byte autocorrelation, lags 1..MAX_LAG (standard biased ACF estimator)
    x = arr.astype(np.float64)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    acf = []
    for k in range(1, MAX_LAG + 1):
        if denom <= 0.0 or n <= k:
            acf.append(0.0)
        else:
            acf.append(float((xc[:-k] * xc[k:]).sum() / denom))

    # --- bigram chi^2 statistics
    a = arr.astype(np.int64)
    n_cells = 65536.0

    def _uniform_chi2(pair_idx):
        total = pair_idx.size
        if total < 1:
            return 0.0
        _, c = np.unique(pair_idx, return_counts=True)
        expected = total / n_cells
        # sum_all (O-E)^2/E  ==  sum_all O^2/E - total   (only observed cells have O > 0)
        return float((c.astype(np.float64) ** 2).sum() / expected - total)

    # (1) overlapping bigrams vs. uniform over 65536 cells
    chi2_overlap = _uniform_chi2(a[:-1] * 256 + a[1:])
    # (2) non-overlapping bigrams vs. uniform (independent samples)
    m = (n // 2) * 2
    chi2_nonoverlap = _uniform_chi2(a[:m:2] * 256 + a[1:m:2])
    # (3) independence of consecutive bytes: observed pair counts vs. product of marginals
    total = n - 1
    if total >= 1:
        first, second = a[:-1], a[1:]
        row = np.bincount(first, minlength=256).astype(np.float64)
        col = np.bincount(second, minlength=256).astype(np.float64)
        cells, c = np.unique(first * 256 + second, return_counts=True)
        expected = row[cells // 256] * col[cells % 256] / total
        chi2_indep = float((c.astype(np.float64) ** 2 / expected).sum() - total)
    else:
        chi2_indep = 0.0

    names = [f'g4_acf_lag_{k:02d}' for k in range(1, MAX_LAG + 1)]
    names += ['g4_bigram_chi2_overlapping', 'g4_bigram_chi2_nonoverlapping', 'g4_bigram_chi2_independence']
    return names, acf + [chi2_overlap, chi2_nonoverlap, chi2_indep]


# ====================================================================== G5
def _blocks(arr, b):
    n_blocks = arr.size // b
    return arr[:n_blocks * b].reshape(n_blocks, b)


def _repeat_stats(blocks):
    """(number of blocks whose value occurs more than once, number of distinct blocks)"""
    if len(blocks) == 0:
        return 0, 0
    uniq, counts = np.unique(blocks, axis=0, return_counts=True)
    return int(counts[counts > 1].sum()), int(len(uniq))


def _mean_hamming(blocks):
    if len(blocks) < 2:
        return 0.0
    x = np.bitwise_xor(blocks[1:], blocks[:-1])
    return float(_POPCOUNT[x].sum(axis=1).mean())


def _infer_block_size(arr):
    """Heuristic block-size inference from repeated aligned blocks (ECB-style leakage).

    For every candidate size we compute the fraction of aligned blocks that repeat.
    Random-looking ciphertext (CBC / CTR / stream ciphers) has none -> 0 ("no block structure found").
    Otherwise the largest candidate whose repeat fraction is at least half of that of the smallest
    repeating candidate is returned.
    """
    frac = {}
    for b in INFER_CANDIDATES:
        blocks = _blocks(arr, b)
        if len(blocks) < 2:
            frac[b] = 0.0
        else:
            repeated, _ = _repeat_stats(blocks)
            frac[b] = repeated / len(blocks)
    repeating = [b for b in INFER_CANDIDATES if frac[b] > 0]
    if not repeating:
        return 0
    ref = frac[repeating[0]]
    return max(b for b in repeating if frac[b] >= 0.5 * ref)


def _g5(arr):
    names, values = [], []
    for b in BLOCK_SIZES:
        blocks = _blocks(arr, b)
        repeated, distinct = _repeat_stats(blocks)
        names += [f'g5_b{b}_repeated_blocks', f'g5_b{b}_distinct_blocks', f'g5_b{b}_mean_hamming']
        values += [repeated, distinct, _mean_hamming(blocks)]
    names.append('g5_inferred_block_size')
    values.append(_infer_block_size(arr))
    return names, values


# ====================================================================== public API
def _compute(raw, groups):
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = arr.size
    if n < 2:
        raise ValueError("ciphertext must contain at least 2 bytes")
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. Available: {list(GROUPS)}")

    all_names, all_values = [], []
    for g in GROUPS:  # fixed canonical order, regardless of the order in `groups`
        if g not in groups:
            continue
        if g == 'G1':
            names, values = _g1(arr, n)
        elif g == 'G2':
            names, values = _g2(arr, raw, n)
        elif g == 'G3':
            names, values = _g3(arr)
        elif g == 'G4':
            names, values = _g4(arr, n)
        else:
            names, values = _g5(arr)
        all_names += names
        all_values += values
    return all_names, np.asarray(all_values, dtype=np.float64)


def extract_features(raw, groups=GROUPS):
    """Feature vector (float32) of one ciphertext given as `bytes`."""
    _, values = _compute(bytes(raw), tuple(groups))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@lru_cache(maxsize=None)
def _names_cached(groups):
    probe = bytes((i * 37 + 11) % 251 for i in range(1024))  # any fixed non-trivial input
    return tuple(_compute(probe, groups)[0])


def get_feature_names(groups=GROUPS):
    """Names of the features, in the same order as `extract_features`."""
    return list(_names_cached(tuple(g for g in GROUPS if g in groups)))
