"""
forest_landscapes.py

Generic "generalized landscape" machinery that works for any forest-like
object which provides:

- forest.point_cloud: np.ndarray of shape (n_points, dim)
- forest.barcode: iterable of bar objects, where each bar has
      bar.birth: float
      bar.death: float (can be math.inf)
      bar.cycle_reps: iterable of cycle representatives

- each cycle representative has
      rep.active_start: float
      rep.active_end: float

The *only* object-specific part is how we turn a cycle representative
into a number. This is supplied by the user as:

    cycle_value_func(rep, point_cloud) -> float

For LoopForest, `rep` is a Loop.
For PersistenceForest, `rep` is a SignedChain with `signed_simplices`.

"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Optional, Tuple, Dict, Union, Literal,List
from numpy.typing import NDArray
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
import math


CycleValueFunc = Callable[[Any, np.ndarray], float]
"""Takes (cycle_rep, point_cloud) and returns a scalar feature."""


@dataclass
class StepFunctionData:
    """
    Representation of a piecewise-constant function:

    f(t) = vals[i]  on [starts[i], ends[i]]
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class PiecewiseLinearFunction:
    """
    Piecewise-linear function specified by breakpoints (xs, ys).
    Between xs[i] and xs[i+1] the function is linear. Outside [xs[0], xs[-1]]
    the function evaluates to 0.0 by default.
    """
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __call__(self, x: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]:
        if self.xs.size == 0:
            if np.isscalar(x):
                return 0.0
            x_arr = np.asarray(x, dtype=float)
            return np.zeros_like(x_arr)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.interp(x_arr, self.xs, self.ys, left=0.0, right=0.0)

        if np.isscalar(x):
            # np.interp returns a scalar np.ndarray for scalar input
            return float(y_arr)
        return y_arr

@dataclass
class GeneralizedLandscapeFamily:
    """
    Container for a whole family of generalized landscapes for a given LoopForest
    and a fixed cycle function.

    - bar_kernels: per-bar kernels (typically already rescaled if mode="pyramid")
    - landscapes: k -> λ_k (k-th landscape) as PiecewiseLinearFunction
    """
    forest_id: str
    func_name: str
    rescaling: str  # e.g. "raw" or "pyramid"
    x_grid: NDArray[np.float64]
    bar_kernels: Dict[int, PiecewiseLinearFunction]
    landscapes: Dict[int, PiecewiseLinearFunction]
    extra_meta: Dict[str, object] = field(default_factory=dict)
    # Optional back-reference for interactive use; safe to ignore for serialization

@dataclass
class LandscapeVectorizer:
    """
    Turn generalized landscapes of LoopForest objects into fixed-size
    feature vectors suitable for machine learning.

    Usage:

        vec = LandscapeVectorizer(
            cycle_func=func,
            max_k=3,
            num_grid_points=256,
            mode="pyramid",
            min_bar_length=0.05,
        )
        vec.fit(training_forests)
        X_train = vec.transform(training_forests)
        X_test  = vec.transform(test_forests)

    After that, X_train and X_test are standard numpy arrays and can be
    used with scikit-learn, PyTorch, etc.
    """
    cycle_func: CycleValueFunc
    max_k: int = 3
    num_grid_points: int = 256
    mode: Literal["raw", "pyramid"] = "pyramid"
    min_bar_length: float = 0.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # fitted attributes
    x_grid_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, forests: Sequence) -> "LandscapeVectorizer":
        """
        Decide on a common time grid for all forests.

        If t_min / t_max are given, they are used directly; otherwise, they
        are inferred from the barcodes (respecting min_bar_length).
        """
        if len(forests) == 0:
            raise ValueError("fit() received an empty list of forests")

        # 1. Determine global t_min / t_max from barcodes if not fixed
        if self.t_min is None or self.t_max is None:
            births: List[float] = []
            deaths: List[float] = []
            for lf in forests:
                if not hasattr(lf, "barcode"):
                    raise AttributeError("One of the LoopForest objects has no 'barcode'")
                for bar in lf.barcode:
                    length = bar.death - bar.birth
                    if length >= self.min_bar_length:
                        births.append(bar.birth)
                        deaths.append(bar.death)
            if not births or not deaths:
                raise ValueError(
                    "No bars of sufficient length found in any forest while fitting "
                    f"(min_bar_length={self.min_bar_length})."
                )
            t_min = min(births) if self.t_min is None else self.t_min
            t_max = max(deaths) if self.t_max is None else self.t_max
        else:
            t_min, t_max = self.t_min, self.t_max

        if t_max <= t_min:
            raise ValueError(f"t_max must be > t_min, got [{t_min}, {t_max}]")

        # 2. Build fixed grid and store it
        self.x_grid_ = np.linspace(t_min, t_max, self.num_grid_points)
        self.is_fitted_ = True
        return self

    def _vectorise_single(self, forest) -> NDArray[np.float64]:
        """
        Compute the feature vector for a single LoopForest on the fixed grid.
        """
        assert self.x_grid_ is not None

        family = forest.compute_generalized_landscape_family(
            self.cycle_func,
            max_k=self.max_k,
            num_grid_points=self.num_grid_points,
            mode=self.mode,
            min_bar_length=self.min_bar_length,
            x_grid=self.x_grid_,  # enforce consistent grid
        )

        # Flatten [k, x] into one vector in a fixed, documented order:
        # [λ_1(x_1..x_N), λ_2(x_1..x_N), ..., λ_max_k(x_1..x_N)].
        pieces: List[NDArray[np.float64]] = []
        for k in range(1, self.max_k + 1):
            if k in family.landscapes:
                ys = family.landscapes[k].ys
            else:
                # If this forest has fewer bars than k, define λ_k ≡ 0
                ys = np.zeros_like(self.x_grid_)
            pieces.append(ys)

        vec = np.concatenate(pieces, axis=0)  # shape: (max_k * num_grid_points,)
        return vec

    def transform(self, forests: Sequence) -> NDArray[np.float64]:
        """
        Transform a sequence of LoopForest objects into a design matrix X.

        X has shape (n_forests, max_k * num_grid_points).
        """
        if not self.is_fitted_ or self.x_grid_ is None:
            raise RuntimeError("LandscapeVectorizer must be fitted before calling transform().")

        n = len(forests)
        if n == 0:
            raise ValueError("transform() received an empty list of forests")

        X = np.zeros((n, self.max_k * self.num_grid_points), dtype=float)
        for i, lf in enumerate(forests):
            X[i, :] = self._vectorise_single(lf)

        return X

    def fit_transform(self, forests: Sequence) -> NDArray[np.float64]:
        """
        Convenience method: fit the grid from forests and return X in one call.
        """
        self.fit(forests)
        return self.transform(forests)

@dataclass
class MultiLandscapeVectorizer:
    """
    Vectorise generalized landscapes for multiple path functions into a single
    feature vector suitable for machine learning.

    For each forest and each path function f_j, we:
      - compute landscapes λ_1,...,λ_max_k on a shared x_grid,
      - flatten the sampled values,
      - optionally append L1 and L2 norms of each λ_k.

    Resulting feature vector (conceptually):

        [ samples for f_1 | stats for f_1 | samples for f_2 | stats for f_2 | ... ]

    Usage:
        vec = MultiLandscapeVectorizer(
            cycle_funcs=[f_const_one, f_length, ...],
            max_k=3,
            num_grid_points=256,
            mode="pyramid",
            min_bar_length=0.05,
            include_stats=True,
        )

        vec.fit(train_forests)
        X_train = vec.transform(train_forests)
        X_test  = vec.transform(test_forests)
    """
    cycle_funcs: List[CycleValueFunc]
    max_k: int = 5
    num_grid_points: int = 256
    mode: Literal["raw", "pyramid"] = "pyramid"
    min_bar_length: float = 0.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    include_stats: bool = False

    # derived
    func_names: Optional[Sequence[str]] = None

    # fitted attributes
    x_grid_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    dx_: Optional[float] = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)
    n_features_: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.func_names is None:
            self.func_names = [
                getattr(f, "__name__", f"func_{i}")
                for i, f in enumerate(self.cycle_funcs)
            ]

    def fit(self, forests: Sequence) -> "MultiLandscapeVectorizer":
        """
        Decide on a common time grid for all forests.

        If t_min / t_max are given, they are used directly; otherwise, they
        are inferred from the barcodes (respecting min_bar_length).
        """
        if len(forests) == 0:
            raise ValueError("fit() received an empty list of forests")

        # 1. Determine global t_min / t_max from barcodes if not fixed
        if self.t_min is None or self.t_max is None:
            births: List[float] = []
            deaths: List[float] = []
            for lf in forests:
                if not hasattr(lf, "barcode"):
                    raise AttributeError("One of the LoopForest objects has no 'barcode'")
                for bar in lf.barcode:
                    length = bar.death - bar.birth
                    if length >= self.min_bar_length:
                        births.append(bar.birth)
                        deaths.append(bar.death)
            if not births or not deaths:
                raise ValueError(
                    "No bars of sufficient length found in any forest while fitting "
                    f"(min_bar_length={self.min_bar_length})."
                )
            t_min = min(births) if self.t_min is None else self.t_min
            t_max = max(deaths) if self.t_max is None else self.t_max
        else:
            t_min, t_max = self.t_min, self.t_max

        if t_max <= t_min:
            raise ValueError(f"t_max must be > t_min, got [{t_min}, {t_max}]")

        # 2. Build fixed grid and store it
        self.x_grid_ = np.linspace(t_min, t_max, self.num_grid_points)
        self.dx_ = float(self.x_grid_[1] - self.x_grid_[0])
        self.is_fitted_ = True

        # 3. Precompute feature dimension
        n_funcs = len(self.cycle_funcs)
        n_samples_per_func = self.max_k * self.num_grid_points
        n_stats_per_func = 0
        if self.include_stats:
            # L1 and L2 per level => 2 * max_k
            n_stats_per_func = 2 * self.max_k
        self.n_features_ = n_funcs * (n_samples_per_func + n_stats_per_func)

        return self

    def _vectorise_single_for_func(
        self,
        forest,
        f: CycleValueFunc,
    ) -> NDArray[np.float64]:
        """
        Compute the feature block for one forest and one path function f.
        Block = [samples, (optional) stats].
        """
        assert self.x_grid_ is not None and self.dx_ is not None

        family = forest.compute_generalized_landscape_family(
            cycle_func=f,
            max_k=self.max_k,
            num_grid_points=self.num_grid_points,
            mode=self.mode,
            min_bar_length=self.min_bar_length,
            x_grid=self.x_grid_,  # enforce consistent grid
            cache=False, # Do not save family to forest
        )

        # Sample part: [λ_1(x_1..x_N), ..., λ_max_k(x_1..x_N)]
        pieces: List[NDArray[np.float64]] = []
        stats: List[float] = []

        for k in range(1, self.max_k + 1):
            if k in family.landscapes:
                ys = family.landscapes[k].ys
            else:
                ys = np.zeros_like(self.x_grid_)
            pieces.append(ys)

            if self.include_stats:
                # L1 norm ~ ∫ |λ_k| dx
                l1 = float(np.sum(np.abs(ys)) * self.dx_)
                # L2 norm ~ (∫ λ_k^2 dx)^(1/2)
                l2 = float(np.sqrt(np.sum(ys ** 2) * self.dx_))
                stats.extend([l1, l2])

        samples_flat = np.concatenate(pieces, axis=0)  # size = max_k * num_grid_points
        if self.include_stats:
            stats_arr = np.asarray(stats, dtype=float)  # size = 2 * max_k
            return np.concatenate([samples_flat, stats_arr], axis=0)
        else:
            return samples_flat

    def transform(self, forests: Sequence) -> NDArray[np.float64]:
        """
        Transform a sequence of LoopForest objects into a design matrix X.

        X has shape (n_forests, n_features_).
        """
        if not self.is_fitted_ or self.x_grid_ is None:
            raise RuntimeError("MultiLandscapeVectorizer must be fitted before calling transform().")

        n = len(forests)
        if n == 0:
            raise ValueError("transform() received an empty list of forests")

        if self.n_features_ is None:
            raise RuntimeError("n_features_ not initialised. Call fit() first.")

        X = np.zeros((n, self.n_features_), dtype=float)

        for i, lf in enumerate(forests):
            blocks: List[NDArray[np.float64]] = []
            for f in self.cycle_funcs:
                block = self._vectorise_single_for_func(lf, f)
                blocks.append(block)
            X[i, :] = np.concatenate(blocks, axis=0)

        return X

    def fit_transform(self, forests: Sequence) -> NDArray[np.float64]:
        """
        Convenience method: fit the grid from forests and return X in one call.
        """
        self.fit(forests)
        return self.transform(forests)


def _build_step_function_data(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        baseline: float = 0.0,
    ) -> StepFunctionData:
        """
        Build the piecewise-constant function associated to a single bar:

        For each cycle representative 'cycle' in bar.cycle_reps, we create an interval
        [cycle.active_start, cycle.active_end] on which the function takes value
        cycle_func(cycle,point_cloud).

        Returns a StepFunctionData object.
        """
        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        starts = np.array(
            [cycle.active_start for cycle in bar.cycle_reps],
            dtype=float,
        )
        ends = np.array(
            [cycle.active_end for cycle in bar.cycle_reps],
            dtype=float,
        )
        vals = np.array(
            [cycle_func(cycle, forest.point_cloud) for cycle in bar.cycle_reps],
            dtype=float,
        )

        if starts.size == 0:
            # Degenerate: no representatives
            domain = (float(bar.birth), float(bar.death))
        else:
            domain = (float(starts.min()), float(ends.max()))

        return StepFunctionData(
            starts=starts,
            ends=ends,
            vals=vals,
            baseline=baseline,
            domain=domain,
            metadata={
                "bar_birth": bar.birth,
                "bar_death": bar.death,
                "root_id": getattr(bar, "root_id", None),
            },
        )

def _build_convolution_with_indicator(
            starts: List[float],
            ends: List[float],
            vals: List[float],
            a: float,
            b: float,
            *,
            tol: float = 1e-12
    ) -> Tuple[Callable[[float], float], List[float], List[float]]:
        """
        Compute h(x) = (f * 1_[a,b])(x) where:
        - f is piecewise-constant: f(t) = vals[i] on [starts[i], ends[i]] and 0 elsewhere
        - 1_[a,b] is the indicator of [a, b] (assumes a <= b)

        Returns:
        (h, xs, ys)
            h  : Callable that evaluates the convolution in O(log n) time via linear interpolation.
            xs : Sorted x 'knot' locations where the slope can change (event points).
            ys : The exact h(x) values at those knots (piecewise-linear nodes).

        Correctness sketch:
        h'(x) = f(x-a) - f(x-b). Each interval [s,e] of f contributes slope jumps of ±v
        at x ∈ {a+s, a+e, b+s, b+e}. Between events, h has constant slope, hence is linear.
        We start from 0 at the left boundary (the sliding window is fully left of support).

        Runtime:
        Building events: O(n)
        Sorting events:  O(n log n) with at most 4n unique points
        One sweep:       O(n)
        Evaluating h(x): O(log n) per query (binary search on xs)
        """
        if b < a:
            raise ValueError("Require a <= b for the indicator [a,b].")
        if not (len(starts) == len(ends) == len(vals)):
            raise ValueError("starts, ends, vals must have equal length.")
        n = len(starts)
        if n == 0:
            # No support: convolution is identically zero
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        # Helper to merge nearly identical event keys to improve numerical stability
        def _quantize(x: float) -> float:
            if tol <= 0:
                return x
            # snap to nearest multiple of tol
            return round(x / tol) * tol

        # Build slope-change "events"
        events = {}  # x -> delta_slope
        def add_event(x: float, delta: float):
            qx = _quantize(x)
            events[qx] = events.get(qx, 0.0) + float(delta)

        for s, e, v in zip(starts, ends, vals):
            if e < s:
                raise ValueError(f"Encountered interval with end < start: [{s}, {e}].")
            if abs(v) < tol or abs(e - s) < tol:
                # Zero value or degenerate interval contributes nothing
                continue
            add_event(a + s, +v)
            add_event(a + e, -v)
            add_event(b + s, -v)
            add_event(b + e, +v)

        if not events:
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        xs = sorted(events.keys())
        # We know h(x)=0 for x < xs[0] (the window [x-b, x-a] is fully left of f’s support).
        ys: List[float] = [0.0]

        # Sweep: on (xs[i], xs[i+1]) the slope is constant; update slope at the left endpoint.
        slope = 0.0
        slopes_per_interval: List[float] = []
        for i in range(len(xs) - 1):
            x_i, x_next = xs[i], xs[i + 1]
            slope += events[x_i]              # slope just to the right of x_i
            slopes_per_interval.append(slope) # slope on (x_i, x_{i+1})
            y_next = ys[-1] + slope * (x_next - x_i)
            ys.append(y_next)

        # Consume the last event to bring slope back (should return to ~0)
        slope += events[xs[-1]]
        # Optional check (tolerant to rounding)
        if not math.isclose(slope, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            # Not fatal; tiny residue can appear from floating noise
            pass

        # Build a fast evaluator by linear interpolation on the piecewise-linear segments
        def h(x: float) -> float:
            if not xs:
                return 0.0
            if x <= xs[0] or x >= xs[-1]:
                return 0.0
            i = bisect_right(xs, x) - 1
            # segment i is (xs[i], xs[i+1]) with slope slopes_per_interval[i]
            return ys[i] + slopes_per_interval[i] * (x - xs[i])

        return h, xs, ys

def compute_convolution_kernel_for_bar(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        tol: float = 1e-12,
    ) -> PiecewiseLinearFunction:
        """
        Use the existing build_convolution_with_indicator to compute

            g(x) = (f * 1_[birth, death])(x)

        where f is the piecewise-constant function from build_step_function_data.
        Returns g as a PiecewiseLinearFunction.
        """
        sf = _build_step_function_data(forest=forest,bar=bar, cycle_func=cycle_func, baseline=0.0)

        a, b = bar.birth, bar.death
        h, xs, ys = _build_convolution_with_indicator(
            sf.starts.tolist(),
            sf.ends.tolist(),
            sf.vals.tolist(),
            a,
            b,
            tol=tol,
        )

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)

        if xs_arr.size > 1:
            domain = (float(xs_arr[0]), float(xs_arr[-1]))
        else:
            domain = (float(a), float(b))

        return PiecewiseLinearFunction(
            xs=xs_arr,
            ys=ys_arr,
            domain=domain,
            metadata={
                **sf.metadata,
                ""
                "func_name": getattr(cycle_func, "__name__", "anonymous"),
                "kernel_type": "raw_convolution",
            },
        )

def compute_generalized_interval_landscape(
        forest,
        cycle_func: CycleValueFunc,
        bar,
    ):
        """
        Helper that computes the (raw) convolution kernel g
        for a single bar and returns (h, xs, ys) as before, but implemented via
        the new PiecewiseLinearFunction.

        Note: this uses the *raw* convolution (no pyramid rescaling).
        For landscapes, prefer compute_generalized_landscape_family.
        """
        if bar is None:
            bar = forest.max_bar()

        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        kernel = compute_convolution_kernel_for_bar(
            forest=forest,
            bar=bar,
            cycle_func=cycle_func,
        )

        xs = kernel.xs.tolist()
        ys = kernel.ys.tolist()

        def h(x: float) -> float:
            return float(kernel(x))

        return h, xs, ys

def compute_landscape_kernel_for_bar(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        mode: Literal["raw", "pyramid"] = "pyramid",
        tol: float = 1e-12,
    ) -> PiecewiseLinearFunction:
        """
        Build the kernel used for landscapes for a single bar.

        - mode="raw":      return g(x) = (f * 1_[birth,death])(x)
        - mode="pyramid":  return λ(x) = 1/2 * g(2x)

        The "pyramid" mode should reproduce the usual persistence landscape
        shape when f ≡ 1.
        """
        raw_kernel = compute_convolution_kernel_for_bar(
            forest = forest,
            bar=bar,
            cycle_func=cycle_func,
            tol=tol,
        )

        if mode == "raw":
            # Just return a shallow copy with explicit metadata
            return PiecewiseLinearFunction(
                xs=raw_kernel.xs.copy(),
                ys=raw_kernel.ys.copy(),
                domain=raw_kernel.domain,
                metadata={
                    **raw_kernel.metadata,
                    "kernel_type": "raw_convolution",
                },
            )

        # mode == "pyramid"
        xs = raw_kernel.xs
        ys = raw_kernel.ys

        # λ(x) = 1/2 * g(2x)
        xs_scaled = xs / 2.0
        ys_scaled = ys / 2.0

        domain = (float(xs_scaled[0]), float(xs_scaled[-1])) if xs_scaled.size > 1 else raw_kernel.domain

        return PiecewiseLinearFunction(
            xs=xs_scaled,
            ys=ys_scaled,
            domain=domain,
            metadata={
                **raw_kernel.metadata,
                "kernel_type": "pyramid_rescaled",
                "rescaling_formula": "lambda(x) = 0.5 * g(2x)",
            },
        )

def compute_generalized_landscape_family(
        forest,
        cycle_func: CycleValueFunc,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        label: Optional[str] = None,
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
    ) -> GeneralizedLandscapeFamily:
        """
        Compute the generalized landscape family for this LoopForest for a given
        cycle_func.

        If x_grid is provided, all landscapes are evaluated on that grid
        (and num_grid_points is ignored). This is crucial for consistent
        vectorisations across multiple LoopForest objects.

        Parameters
        ----------
        cycle_func:
            Function f(SignedChain,point_cloud) -> scalar.
        max_k:
            Number of landscapes λ_1, ..., λ_max_k to compute.
        num_grid_points:
            Number of grid points used when x_grid is None.
        mode:
            "raw" or "pyramid" (see compute_landscape_kernel_for_bar).
        label:
            Key used to store the family in forest.landscape_families.
        min_bar_length:
            Ignore bars with (death - birth) < min_bar_length.
        x_grid:
            Optional fixed grid on which to evaluate the landscapes.
        cache:
            Decides if landscapes is saved to forest or only returned.
        """
        if not hasattr(forest, "barcode"):
            raise AttributeError("LoopForest has no 'barcode' attribute. Did you compute it?")

        # 1. Filter bars by length
        bars = [
            bar for bar in forest.barcode
            if (bar.death - bar.birth) >= min_bar_length
        ]

        if not bars:
            raise ValueError(
                f"No bars with length >= {min_bar_length}. "
                "Increase min_bar_length or check your barcode."
            )

        # 2. Compute kernels for each bar
        bar_kernels: Dict[int, PiecewiseLinearFunction] = {}
        global_min_x = float("inf")
        global_max_x = float("-inf")

        for i, bar in enumerate(bars):
            kernel = compute_landscape_kernel_for_bar(
                forest,
                bar,
                cycle_func,
                mode=mode,
            )
            bar_kernels[i] = kernel

            global_min_x = min(global_min_x, kernel.domain[0])
            global_max_x = max(global_max_x, kernel.domain[1])

        if not np.isfinite(global_min_x) or not np.isfinite(global_max_x):
            raise RuntimeError("Failed to infer a finite global domain for the kernels.")

        if global_max_x <= global_min_x:
            raise RuntimeError(
                f"Non-positive domain width: [{global_min_x}, {global_max_x}]"
            )

        # 3. Common grid
        if x_grid is None:
            x_grid = np.linspace(global_min_x, global_max_x, num_grid_points)
        else:
            x_grid = np.asarray(x_grid, dtype=float)
            if x_grid.ndim != 1 or x_grid.size < 2:
                raise ValueError("x_grid must be a 1D array with at least 2 points")
        num_grid_points = x_grid.size  # ensure consistency

        # 4. Evaluate all kernels on the grid
        n_bars = len(bars)
        values = np.zeros((n_bars, num_grid_points), dtype=float)
        for i, kernel in bar_kernels.items():
            values[i, :] = kernel(x_grid)

        # 5. Compute order statistics along the bar axis
        # sorted_vals[k, j] = (k+1)-th largest value at x_grid[j]
        sorted_vals = np.sort(values, axis=0)[::-1, :]  # descending along axis 0
        max_possible_k = sorted_vals.shape[0]

        landscapes: Dict[int, PiecewiseLinearFunction] = {}
        for k in range(1, max_k + 1):
            if k <= max_possible_k:
                y_k = sorted_vals[k - 1, :]
            else:
                # If we ask for more landscapes than bars, pad with zeros.
                y_k = np.zeros(num_grid_points, dtype=float)

            landscapes[k] = PiecewiseLinearFunction(
                xs=x_grid.copy(),
                ys=y_k.copy(),
                domain=(float(x_grid[0]), float(x_grid[-1])),
                metadata={
                    "k": k,
                    "func_name": getattr(cycle_func, "__name__", "anonymous"),
                    "mode": mode,
                    "min_bar_length": min_bar_length,
                },
            )

        # 6. Assemble family object
        forest_id = getattr(forest, "name", f"Forest@{id(forest)}")

        family = GeneralizedLandscapeFamily(
            forest_id=forest_id,
            func_name=getattr(cycle_func, "__name__", "anonymous"),
            rescaling=mode,
            x_grid=x_grid,
            bar_kernels=bar_kernels,
            landscapes=landscapes,
            extra_meta={
                "num_grid_points": num_grid_points,
                "min_bar_length": min_bar_length,
                "global_min_x": global_min_x,
                "global_max_x": global_max_x,
            },
        )

        # Cache on the LoopForest instance
        if cache:
            if not hasattr(forest, "landscape_families"):
                forest.landscape_families = {}

            key = label or family.func_name
            forest.landscape_families[key] = family

        return family

def plot_landscape_family(
        forest,
        label: str,
        ks: Optional[List[int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        title: Optional[str] = None,
    ):

    if not hasattr(forest, "landscape_families"):
        raise AttributeError("No landscape_families attribute on this LoopForest")

    family = forest.landscape_families[label]

    if ks is None:
        ks = sorted(family.landscapes.keys())

    if ax is None:
        fig, ax = plt.subplots()

    for k in ks:
        plf = family.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=fr"$\lambda_{k}$")

    ax.set_xlabel("filtration value")
    ax.set_ylabel("landscape value")
    if title is None:
        title = f"Generalized landscapes of {label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def plot_landscape_comparison_between_functionals(forest,
    labels: list[str],
    k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
):

    if not hasattr(forest, "landscape_families"):
        raise AttributeError(f"No landscape_families attribute on {forest}")


    families = [forest.landscape_families[label] for label in labels]

    if ax is None:
        fig, ax = plt.subplots()

    for fam, label in zip(families, labels):
        if k not in fam.landscapes:
            continue
        plf = fam.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=label)

    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    if title is None:
        title = fr"Comparison of $\lambda_{k}$"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def plot_landscape_comparison(
    forests: List,
    label: str,
    k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    forest_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    for forest in forests:
        if not hasattr(forest, "landscape_families"):
            raise AttributeError(f"No landscape_families attribute on {forest}")

    families = [forest.landscape_families[label] for forest in forests]

    if ax is None:
        fig, ax = plt.subplots()

    if forest_labels is None:
        forest_labels = [fam.forest_id for fam in families]

    for fam, forest_label in zip(families, forest_labels):
        if k not in fam.landscapes:
            continue
        plf = fam.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=forest_label)

    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    if title is None:
        title = fr"Comparison of {label} $\lambda_{k}$"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
