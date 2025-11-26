from __future__ import annotations
import json, math, random, datetime
from typing import Dict, List, Sequence, Optional, Callable, Any, Iterable
import numpy as np
from matplotlib import colors as mcolors

# ======================
# Color helpers & Lab
# ======================

def _to_hex(c):
    if isinstance(c, str):
        return mcolors.to_hex(mcolors.to_rgb(c))
    return mcolors.to_hex(c)



def _mix(rgb1, rgb2, t):
    return tuple((1 - t) * a + t * b for a, b in zip(rgb1, rgb2))

def _luminance(rgb):
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def _hue_shift(rgb, delta_h):
    h, s, v = mcolors.rgb_to_hsv(rgb)
    h = (h + delta_h) % 1.0
    return tuple(mcolors.hsv_to_rgb((h, s, v)))

# --- sRGB -> Lab (D65) utilities (no external deps) ---

# Reference white (D65)
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

def _srgb_to_linear(c):
    # c in [0,1]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _f_lab(t):
    # CIE standard helper
    delta = 6/29
    return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2) + 4/29)
   

def _rgb_to_xyz(rgb):
    # rgb in [0,1], sRGB, D65
    r, g, b = _srgb_to_linear(np.array(rgb))
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return X, Y, Z


    
def _rgb_to_lab(rgb):
    X, Y, Z = _rgb_to_xyz(rgb)
    fx, fy, fz = _f_lab(X/_Xn), _f_lab(Y/_Yn), _f_lab(Z/_Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.array([L, a, b])

    

def _lab_distance(lab1, lab2):
    # ΔE*76
    d = lab1 - lab2
    return float(np.sqrt(np.dot(d, d)))
 
# ======================
# Base color generator
# ======================

def _hue_sequence(n: int, start: float) -> Iterable[float]:
    """Golden-ratio hue stepping for uniform coverage."""
    phi = 0.61803398875
    h = start
    for _ in range(n):
        h = (h + phi) % 1.0
        yield h

def _generate_candidates(
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.0),
    vals: Sequence[float] = (0.80, 0.95),
    hue_start: float = 0.11,   # near orange by default
    L_bounds: Sequence[float] = (30.0, 92.0),  # keep mid/bright; avoid near-black/white
) -> List[tuple]:
    """
    Returns a list of (hex, lab) candidate colors.
    """
    candidates = []
    for h in _hue_sequence(num_hues, hue_start):
        for s in sats:
            for v in vals:
                rgb = tuple(mcolors.hsv_to_rgb((h, s, v)))
                lab = _rgb_to_lab(rgb)
                if L_bounds[0] <= lab[0] <= L_bounds[1]:
                    candidates.append((_to_hex(rgb), lab))
    # Deduplicate very close colors (rare with golden stepping, but cheap)
    uniq = []
    for hx, lab in candidates:
        if not uniq or _lab_distance(lab, uniq[-1][1]) > 0.5:
            uniq.append((hx, lab))
    return uniq

def distinct_base_colors(
    n_sets: int,
    *,
    seed: Optional[int] = None,
    prefer_start: Optional[str] = None,
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.00),
    vals: Sequence[float] = (0.80, 0.95),
) -> List[str]:
    """
    Pick n_sets bases by farthest-point sampling in Lab from a large HSV candidate pool.
    - 'seed' changes the initial pick (and thus the whole selection).
    - 'prefer_start' (hex or any Matplotlib color) biases the first color near this target.
    - Adjust num_hues/sats/vals to widen or densify the candidate gamut.
    """
    rng = random.Random(seed)
    hue_start = rng.random() if seed is not None else 0.11  # randomize starting phase if seeded
    cand = _generate_candidates(num_hues=num_hues, sats=sats, vals=vals, hue_start=hue_start)
    if len(cand) == 0:
        raise RuntimeError("No color candidates generated; adjust sats/vals/L_bounds.")

    # Choose first color:
    if prefer_start is not None:
        target_lab = _rgb_to_lab(mcolors.to_rgb(prefer_start))
        first_idx = min(range(len(cand)), key=lambda i: _lab_distance(cand[i][1], target_lab))
    else:
        first_idx = rng.randrange(len(cand)) if seed is not None else 0

    selected_idx = [first_idx]
    selected_lab = [cand[first_idx][1]]

    # Precompute distances to speed up greedy FPS
    min_dists = np.array([_lab_distance(c[1], selected_lab[0]) for c in cand], dtype=float)

    while len(selected_idx) < min(n_sets, len(cand)):
        # pick the candidate with the largest distance to the current selected set
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        sel_lab = cand[next_idx][1]
        # update distances
        for i, (hx, lab) in enumerate(cand):
            d = _lab_distance(lab, sel_lab)
            if d < min_dists[i]:
                min_dists[i] = d



    return [cand[ selected_idx[i % len(selected_idx) ] ][0] for i in range(n_sets)]

# ======================
# Within-set variants
# ======================

def variants_for_set(base_hex: str,
                     n: int,
                     tint_strength: float = 0.8,
                     shade_strength: float = 0.7,
                     hue_jitter: float = 0.04,
                     order: str = "light_to_dark") -> List[str]:
    """
    Ordered similar-but-not-identical colors inside one set.
    """
    base_rgb = mcolors.to_rgb(base_hex)
    L = _luminance(base_rgb)
    white, black = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)

    if L < 0.55:
        mixes = np.linspace(0.15, tint_strength, n)
        seq = [_mix(base_rgb, white, t) for t in mixes]
        seq = seq if order == "light_to_dark" else list(reversed(seq))
    elif L > 0.80:
        mixes = np.linspace(0.10, shade_strength, n)
        seq = [_mix(base_rgb, black, t) for t in mixes]
        seq = seq if order == "dark_to_light" else list(reversed(seq))
    else:
        tints  = [_mix(base_rgb, white, t) for t in np.linspace(0.0, tint_strength * 0.6, math.ceil(n/2))]
        shades = [_mix(base_rgb, black, t) for t in np.linspace(0.0, shade_strength * 0.5, n - len(tints))]
        seq = tints + list(reversed(shades))
        if order == "dark_to_light":
            seq = list(reversed(seq))

    if n > 1 and hue_jitter > 0:
        offsets = np.linspace(-hue_jitter, hue_jitter, n)
        seq = [_hue_shift(rgb, dh) for rgb, dh in zip(seq, offsets)]

    return [_to_hex(rgb) for rgb in seq]

# ======================
# Scheme build/save/load
# ======================

def build_color_scheme(set_sizes: Sequence[int],
                       set_ids: Sequence[str],
                       seed: Optional[int] = None,
                       order_within_set: str = "light_to_dark",
                       prefer_start: Optional[str] = None,
                       num_hues: int = 360,
                       sats: Sequence[float] = (0.65, 0.85, 1.00),
                       vals: Sequence[float] = (0.80, 0.95),
                       max_variants_per_set: int = 20) -> Dict[str, Any]:
    if len(set_sizes) != len(set_ids):
        raise ValueError("set_sizes and set_ids must have the same length")
    bases = distinct_base_colors(
        len(set_ids),
        seed=seed,
        prefer_start=prefer_start,
        num_hues=num_hues,
        sats=sats,
        vals=vals,
    )
    sets = {
        sid: {
            "base": base,
            "colors": variants_for_set(base, sz, order=order_within_set)
        }
        for sid, base, sz in zip(set_ids, bases, set_sizes)
    }
    return {
        "_meta": {
            "algorithm": "Lab-FPS over HSV rings",
            "seed": seed,
            "params": {"num_hues": num_hues, "sats": list(sats), "vals": list(vals)},
            "base_colors": bases,
        },
        "sets": sets,
    }


# -----------------------
# <== Your objects entry point
# -----------------------

def _sid(ob) -> int:
    # works for np.int32/64, Python int, etc.
    return int(getattr(ob, "root_id"))

def _stable_sid_key(sid: Any) -> Any:
    """
    Return a stable, immutable key for a set id.
    If sid is already a simple immutable (str/int/tuple), use it.
    Otherwise fall back to id(sid) so it won't change if sid's state mutates.
    """
    if isinstance(sid, (str, int, float, tuple)):
        return sid
    return ("objid", id(sid))

def build_scheme_from_bars(
    bars: Iterable,
    *,
    seed: Optional[int] = None,
    order_within_set: str = "light_to_dark",
    prefer_start: Optional[str] = "#ff7f0e",  # e.g., "#ff7f0e" to bias first base toward orange
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.00),
    vals: Sequence[float] = (0.80, 0.95),
    max_variants_per_set: int = 16,
) -> Dict[str, Any]:
    """
    Build a reusable JSON-friendly scheme straight from bars_set.

    - Groups by ob.color_set
    - Preserves encounter order per group unless 'order_key' is provided
    - Returns the scheme (save to JSON for reuse)
    """
    # Group objects by set
    groups: Dict[Any, List] = {}
    for bar in bars:
        sid = _sid(bar)
        groups.setdefault(_stable_sid_key(sid), []).append(bar)


    # Stable set id order: by first appearance (insertion order of groups)
    set_ids = list(groups.keys())
    set_sizes_capped = [min(len(groups[sid]), max_variants_per_set) for sid in set_ids]


    scheme = build_color_scheme(
        set_sizes=set_sizes_capped,
        set_ids=set_ids,
        seed=seed,
        order_within_set=order_within_set,
        prefer_start=prefer_start,
        num_hues=num_hues,
        sats=sats,
        vals=vals
    )
    return scheme

def color_map_for_bars(
    bars: Iterable,
    seed: Optional[int]=None,
    prefer_start: Optional[str] = "#ff7f0e",
    *,
    by_id: bool = False,
) -> Dict[object, str]:
    """
    Build {ob -> '#RRGGBB'} using an existing scheme.
    - If objects are unhashable, set by_id=True to return {id(ob) -> color}
    """
    scheme = build_scheme_from_bars(
        bars,
        seed=seed,
        prefer_start=prefer_start
    )

    # Recreate groups to align colors with the same within-set order
    groups: Dict[str, List] = {}
    for bar in bars:
        sid = _sid(bar)
        groups.setdefault(_stable_sid_key(sid), []).append(bar)

    color_map = {}
    for sid_key, obs in groups.items():
        colors = list(scheme["sets"][sid_key]["colors"])

        # Always cycle on shortfall
        for i, ob in enumerate(obs):
            color_map[id(ob) if by_id else ob] = colors[i % len(colors)]
    return color_map

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Dummy class for demonstration
    class Bar:
        def __init__(self, name, color_set, rank):
            self.name = name
            self.color_set = color_set
            self.rank = rank
        def __repr__(self):
            return f"Bar({self.name})"

    # Example bars_set
    bars_set = set([
        Bar("a1", "A", 2), Bar("a2", "A", 1), Bar("b1", "B", 3),
        Bar("c1", "C", 1), Bar("b2", "B", 2)
    ])

    # 1) Build & save a reusable scheme from the current bars_set
    scheme = build_scheme_from_bars(
        bars_set,
        seed=123,
    )

    color_map = color_map_for_bars(
        bars_set,
        by_id=False,  # set True if your objects are unhashable
    )

    # color_map is {Bar(...) -> "#RRGGBB"} usable in plotting
    # Example with Matplotlib:
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(10)
    for ob in bars_set:
        y = np.sin(x * (ob.rank + 1) / 4)
        plt.plot(x, y, label=ob.name, color=color_map[ob])
    plt.legend()
    plt.show()
    """