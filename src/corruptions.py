"""Corruption physics.

The low-light observation model has two stages:

    stage 1 (analog):      I' = a * I + eta,        eta ~ N(0, sigma^2)
    stage 2 (digitization): Q = clip(round(I'), 0, 255)  ->  uint8

Stage 1 is invertible in principle (multiply by 1/a). Stage 2 is not: values
below 0.5 map to 0, adjacent levels collapse, and the noise gets baked in.
Several experiments depend on separating the two, so every function here is
explicit about which stage it applies.
"""

import hashlib

import numpy as np

BRIGHTNESS_FACTORS = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
NOISE_STD = [0.0, 2.0, 4.0, 6.0, 9.0, 13.0]


def _deterministic_noise(image: np.ndarray, severity: int, sigma: float) -> np.ndarray:
    """Order-independent Gaussian noise keyed on (image bytes, severity).

    Why not np.random.global? Because the noise draw then depends on HOW MANY
    corruptions happened before this one in the process, so two scripts that
    evaluate the same (image, severity) pair with different call orders get
    different images -- and 'same seed, same result' silently breaks. This
    generator derives a reproducible stream from the image content itself:
    identical inputs give identical noise regardless of call order, and no
    global RNG state is consumed or mutated.
    """
    key = hashlib.blake2b(image.tobytes() + bytes([severity]), digest_size=16)
    seed = int.from_bytes(key.digest(), "little") % (2**32)
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, sigma, image.shape).astype(np.float32)


def low_light_stage1(image: np.ndarray, severity: int,
                     factors=None, noise_std=None) -> np.ndarray:
    """Stage 1 only: analog attenuation + noise. Returns float32 in [0, 255].

    No uint8 cast, so no irreversible quantization. Used as the 'recoverable'
    arm in the quantization experiment and as ground truth for restoration
    baselines.
    """
    factors = factors or BRIGHTNESS_FACTORS
    noise_std = noise_std or NOISE_STD
    img = image.astype(np.float32) * factors[severity]
    if noise_std[severity] > 0:
        # float32 noise: normal() returns float64, and float32 + float64 would
        # silently promote the pipeline to float64 (see quantization experiment).
        # Noise is order-independent: same (image, severity) -> same draw.
        img = img + _deterministic_noise(image, severity, noise_std[severity])
    return np.clip(img, 0, 255).astype(np.float32)


def low_light(image: np.ndarray, severity: int,
              factors=None, noise_std=None) -> np.ndarray:
    """Full two-stage corruption (the standard ladder used across the repo).

    Matches the original notebooks bit-for-bit given the same RNG state.
    """
    return low_light_stage1(image, severity, factors, noise_std).astype(np.uint8)


def quantization_gap_truth(image: np.ndarray, severity: int,
                           noise_free: bool = True) -> float:
    """RMS between stage-1 (float) and stage-2 (uint8 round-trip).

    This is the mean-squared-error cost of digitization itself, independent of
    any model. By default uses the NOISE-FREE ladder so the gap isolates pure
    rounding/clip error; with noise, the function would re-draw eta and measure
    noise+quantization jointly.
    Reported alongside the accuracy gap in the quantization table.
    """
    if noise_free:
        factors = list(BRIGHTNESS_FACTORS)
        s1 = np.clip(image.astype(np.float32) * factors[severity], 0, 255)
        s2 = np.clip(np.round(s1), 0, 255).astype(np.uint8).astype(np.float32)
        return float(np.sqrt(np.mean((s1 - s2) ** 2)))
    s1 = low_light_stage1(image, severity)
    s2 = s1.astype(np.uint8).astype(np.float32)
    return float(np.sqrt(np.mean((s1 - s2) ** 2)))


# ---------------------------------------------------------------------------
# Band-limited controls (frequency experiment)
# ---------------------------------------------------------------------------

def _fft_filter(image: np.ndarray, cutoff_frac: float, keep: str) -> np.ndarray:
    img = image.astype(np.float32)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    radius = cutoff_frac * max_dist
    mask = (dist <= radius) if keep == "low" else (dist > radius)
    for ch in range(img.shape[2]):
        f = np.fft.fftshift(np.fft.fft2(img[:, :, ch]))
        out[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    return np.clip(out, 0, 255).astype(np.uint8)


def low_pass(image: np.ndarray, severity: int) -> np.ndarray:
    cutoffs = [0.9, 0.7, 0.5, 0.35, 0.2]
    return _fft_filter(image, cutoffs[severity - 1], keep="low")


def high_pass(image: np.ndarray, severity: int) -> np.ndarray:
    cutoffs = [0.05, 0.1, 0.15, 0.2, 0.3]
    return _fft_filter(image, cutoffs[severity - 1], keep="high")


# ---------------------------------------------------------------------------
# Classical mitigation (zero-training baselines)
# ---------------------------------------------------------------------------

def simple_gain(image: np.ndarray, severity: int) -> np.ndarray:
    """Undo attenuation: I/a. Recovers contrast, amplifies noise equally."""
    img = image.astype(np.float32) / BRIGHTNESS_FACTORS[severity]
    return np.clip(img, 0, 255).astype(np.uint8)


def gamma_correct(image: np.ndarray, severity: int, base: float = 0.35) -> np.ndarray:
    """Perceptual brightening: exponent = 1 + base*severity.

    Amplifies shadows more than highlights (unlike linear gain), which is the
    classical low-light enhancement shape.
    """
    exponent = 1.0 + base * max(severity, 1)
    x = np.asarray(image).astype(np.float32) / 255.0
    out = np.power(x, 1.0 / exponent) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_enhance(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """CLAHE on the LAB L-channel — the standard zero-training enhancement."""
    import skimage.color
    import skimage.exposure as skexposure

    img = np.asarray(image).astype(np.float32) / 255.0
    lab = skimage.color.rgb2lab(img)
    lab[:, :, 0] = skexposure.equalize_adapthist(lab[:, :, 0] / 100.0, clip_limit=clip_limit) * 100.0
    return (np.clip(skimage.color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
