"""Hand-crafted statistical features (groups G1-G5) computed from one ciphertext.

G1  Byte distribution      : 256-bin histogram, chi^2 vs uniform, first four moments
G2  Entropy/compressibility: Shannon (Miller-Madow), Renyi-2, min-entropy, zlib & LZMA ratios
G3  Bit-level randomness   : p-values of six NIST SP 800-22 tests
                             (frequency, block frequency, runs, longest run, cumulative sums, approx. entropy)
G4  Serial dependence      : byte autocorrelation at lags 1..32, bigram chi^2 statistics
G5  Block structure        : for b in {8, 16}: repeated aligned blocks, distinct blocks,
                             mean Hamming distance of consecutive blocks; plus an inferred block size

`extract_features(raw_bytes)` returns a 1-D float32 vector, `feature_names()` its column names.
"""
import lzma
import math
import zlib

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm

MAX_LAG = 32
BLOCK_SIZES = (8, 16)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


# ----------------------------------------------------------------------------- G1
def _g1(data, counts):
    n = len(data)
    expected = n / 256.0
    hist = counts / n
    chi2 = float(((counts - expected) ** 2 / expected).sum())

    x = data.astype(np.float64)
    mean = x.mean()
    var = x.var()
    if var > 0:
        std = math.sqrt(var)
        skew = float(((x - mean) ** 3).mean() / std ** 3)
        kurt = float(((x - mean) ** 4).mean() / var ** 2)
    else:
        skew, kurt = 0.0, 0.0
    return list(hist) + [chi2, mean, var, skew, kurt]


# ----------------------------------------------------------------------------- G2
def _g2(raw, data, counts):
    n = len(data)
    p = counts[counts > 0] / n
    k = len(p)
    shannon = float(-(p * np.log2(p)).sum())
    # Miller-Madow bias correction: + (K - 1) / (2 N) nats  (converted to bits)
    shannon_mm = shannon + (k - 1) / (2.0 * n * math.log(2))
    renyi2 = float(-math.log2((p ** 2).sum()))
    min_entropy = float(-math.log2(p.max()))
    zlib_ratio = len(zlib.compress(raw, 9)) / n
    lzma_ratio = len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)) / n
    return [shannon_mm, renyi2, min_entropy, zlib_ratio, lzma_ratio]


# ----------------------------------------------------------------------------- G3 (NIST SP 800-22)
def _bits(data):
    return np.unpackbits(data).astype(np.int8)


def nist_frequency(bits):
    n = len(bits)
    s = np.abs((2 * bits.astype(np.int64) - 1).sum()) / math.sqrt(n)
    return float(erfc(s / math.sqrt(2)))


def nist_block_frequency(bits, m=128):
    n = len(bits)
    n_blocks = n // m
    if n_blocks == 0:
        return 0.0
    pi = bits[:n_blocks * m].reshape(n_blocks, m).mean(axis=1)
    chi2 = 4.0 * m * ((pi - 0.5) ** 2).sum()
    return float(gammaincc(n_blocks / 2.0, chi2 / 2.0))


def nist_runs(bits):
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):  # prerequisite frequency test failed
        return 0.0
    v_obs = 1 + int((bits[1:] != bits[:-1]).sum())
    num = abs(v_obs - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return float(erfc(num / den))


def _longest_run_of_ones(block):
    """block: 2-D array [N, M] of 0/1. Returns the longest run of ones per row."""
    best = np.zeros(block.shape[0], dtype=np.int64)
    cur = np.zeros(block.shape[0], dtype=np.int64)
    for j in range(block.shape[1]):
        cur = (cur + 1) * block[:, j]
        best = np.maximum(best, cur)
    return best


def nist_longest_run(bits):
    n = len(bits)
    if n < 128:
        return 0.0
    if n < 6272:
        m, pis, lo, hi = 8, [0.2148, 0.3672, 0.2305, 0.1875], 1, 4
    else:
        m, pis, lo, hi = 128, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124], 4, 9
    n_blocks = n // m
    runs = _longest_run_of_ones(bits[:n_blocks * m].reshape(n_blocks, m).astype(np.int64))
    v = np.bincount(np.clip(runs, lo, hi) - lo, minlength=hi - lo + 1).astype(np.float64)
    pis = np.array(pis)
    chi2 = (((v - n_blocks * pis) ** 2) / (n_blocks * pis)).sum()
    return float(gammaincc((len(pis) - 1) / 2.0, chi2 / 2.0))


def nist_cumulative_sums(bits, reverse=False):
    n = len(bits)
    x = 2 * bits.astype(np.int64) - 1
    if reverse:
        x = x[::-1]
    z = int(np.abs(np.cumsum(x)).max())
    if z == 0:
        return 0.0
    sn = math.sqrt(n)
    k1 = np.arange(int(math.floor((-n / z + 1) / 4)), int(math.floor((n / z - 1) / 4)) + 1)
    k2 = np.arange(int(math.floor((-n / z - 3) / 4)), int(math.floor((n / z - 1) / 4)) + 1)
    s1 = (norm.cdf((4 * k1 + 1) * z / sn) - norm.cdf((4 * k1 - 1) * z / sn)).sum()
    s2 = (norm.cdf((4 * k2 + 3) * z / sn) - norm.cdf((4 * k2 + 1) * z / sn)).sum()
    return float(min(1.0, max(0.0, 1.0 - s1 + s2)))


def nist_approximate_entropy(bits, m=None):
    n = len(bits)
    if m is None:  # NIST recommends m < floor(log2(n)) - 5
        m = max(1, min(5, int(math.floor(math.log2(n))) - 6))

    def phi(mm):
        ext = np.concatenate([bits, bits[:mm - 1]]).astype(np.int64) if mm > 1 else bits.astype(np.int64)
        idx = np.zeros(n, dtype=np.int64)
        for j in range(mm):
            idx = (idx << 1) | ext[j:j + n]
        c = np.bincount(idx, minlength=2 ** mm) / n
        c = c[c > 0]
        return float((c * np.log(c)).sum())

    apen = phi(m) - phi(m + 1)
    chi2 = 2.0 * n * (math.log(2) - apen)
    return float(gammaincc(2 ** (m - 1), chi2 / 2.0))


def _g3(data):
    bits = _bits(data)
    return [
        nist_frequency(bits),
        nist_block_frequency(bits),
        nist_runs(bits),
        nist_longest_run(bits),
        nist_cumulative_sums(bits, reverse=False),  # forward
        nist_cumulative_sums(bits, reverse=True),   # backward (NIST reports both p-values)
        nist_approximate_entropy(bits),
    ]


# ----------------------------------------------------------------------------- G4
def _g4(data):
    x = data.astype(np.float64)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    acf = []
    for lag in range(1, MAX_LAG + 1):
        if denom > 0 and len(x) > lag:
            acf.append(float((x[:-lag] * x[lag:]).sum() / denom))
        else:
            acf.append(0.0)

    d = data.astype(np.int64)
    # overlapping bigrams
    pairs = d[:-1] * 256 + d[1:]
    # aligned (non-overlapping) bigrams
    m = (len(d) // 2) * 2
    aligned = d[:m:2] * 256 + d[1:m:2]

    def chi2(idx):
        c = np.bincount(idx, minlength=65536).astype(np.float64)
        e = len(idx) / 65536.0
        return float(((c - e) ** 2 / e).sum())

    return acf + [chi2(pairs), chi2(aligned)]


# ----------------------------------------------------------------------------- G5
def _g5(data):
    out, repeated_per_b = [], {}
    for b in BLOCK_SIZES:
        n_blocks = len(data) // b
        if n_blocks == 0:
            out += [0.0, 0.0, 0.0]
            repeated_per_b[b] = 0
            continue
        blocks = data[:n_blocks * b].reshape(n_blocks, b)
        _, counts = np.unique(blocks, axis=0, return_counts=True)
        distinct = len(counts)
        repeated = n_blocks - distinct  # surplus copies of already-seen aligned blocks
        if n_blocks > 1:
            hamming = float(_POPCOUNT[np.bitwise_xor(blocks[1:], blocks[:-1])].sum(axis=1).mean())
        else:
            hamming = 0.0
        out += [float(repeated), float(distinct), hamming]
        repeated_per_b[b] = repeated
    # Inferred block size: the candidate with the most repeated aligned blocks (0 = no repeats seen)
    best_b = max(BLOCK_SIZES, key=lambda b: repeated_per_b[b])
    inferred = float(best_b) if repeated_per_b[best_b] > 0 else 0.0
    return out + [inferred]


# ----------------------------------------------------------------------------- public API
def extract_features(raw):
    """raw: bytes of one ciphertext -> float32 vector of length len(feature_names())."""
    data = np.frombuffer(raw, dtype=np.uint8)
    counts = np.bincount(data, minlength=256).astype(np.float64)
    vec = _g1(data, counts) + _g2(raw, data, counts) + _g3(data) + _g4(data) + _g5(data)
    vec = np.nan_to_num(np.asarray(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return vec.astype(np.float32)


def feature_names():
    names = [f"g1_hist_{i}" for i in range(256)]
    names += ["g1_chi2_uniform", "g1_mean", "g1_variance", "g1_skewness", "g1_kurtosis"]
    names += ["g2_shannon_miller_madow", "g2_renyi2", "g2_min_entropy", "g2_zlib_ratio", "g2_lzma_ratio"]
    names += ["g3_nist_frequency", "g3_nist_block_frequency", "g3_nist_runs", "g3_nist_longest_run",
              "g3_nist_cusum_forward", "g3_nist_cusum_backward", "g3_nist_approx_entropy"]
    names += [f"g4_autocorr_lag{k}" for k in range(1, MAX_LAG + 1)]
    names += ["g4_bigram_chi2_overlapping", "g4_bigram_chi2_aligned"]
    for b in BLOCK_SIZES:
        names += [f"g5_b{b}_repeated_blocks", f"g5_b{b}_distinct_blocks", f"g5_b{b}_mean_hamming"]
    names += ["g5_inferred_block_size"]
    return names


NUM_FEATURES = len(feature_names())
