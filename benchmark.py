"""
Benchmark script for PersistenceForest(point_cloud).

This script measures the construction time of PersistenceForest for
different point-cloud sampling schemes and point counts.

Output
------
A CSV file (default: `persistence_forest_benchmark.csv`) with columns:

    sampler                     - name of the sampling method
    n_points                    - number of points in the point cloud
    run_index                   - index of the repetition (0..repeats-1)
    dim                         - ambient dimension of the point cloud
    reduce                      - 1 if reduction was used, 0 otherwise
    compute_barcode             - 1 if barcodes were computed, 0 otherwise
    time_point_cloud_s          - time to generate the point cloud
    time_persistence_forest_s   - time to construct PersistenceForest
    time_total_s                - total time (generation + forest)
    seed                        - random seed used for this run

This format is suitable for direct use in Python/R/Julia for plotting
scaling curves in your paper.
"""

from __future__ import annotations


import csv
import os
import time
from typing import Callable, Dict, Iterable, Sequence, Optional
import statistics
import matplotlib.pyplot as plt

import numpy as np

# Local imports
from PersistenceForest import PersistenceForest
from point_cloud_sampling import (
    sample_noisy_circle,
    sample_uniform_points,
    sample_points_without_balls,
    sample_noisy_sphere,
)

# ---------------------------------------------------------------------------
# Sampling configuration
# ---------------------------------------------------------------------------

Sampler = Callable[[int, int], np.ndarray]


def _make_samplers() -> Dict[str, Sampler]:
    """Return a dictionary mapping method names to sampling callables.

    Each callable has the signature (n_points, seed) -> point_cloud (n,d).
    """
    return {
        "noisy_circle_noise-std-dot05": lambda n, seed: sample_noisy_circle(n, seed=seed, noise_std=0.05),
        "noisy_2sphere_noise-std-dot05": lambda n, seed: sample_noisy_sphere(n, dim =3, seed=seed, noise_std=0.05),
        "uniform_2D": lambda n, seed: sample_uniform_points(n,dim=2, seed=seed),
        "uniform_3D": lambda n, seed: sample_uniform_points(n,dim=3, seed=seed),
        "uniform_2D_with_30holes_radius-max-dot05": lambda n, seed: sample_points_without_balls(n=n,dim=2,seed=seed,radius_range=[0,0.5],num_discs=30),
        "uniform_3D_with_30holes_radius-max-dot05": lambda n, seed: sample_points_without_balls(n=n,dim=3,seed=seed,radius_range=[0,0.5],num_discs=30)

    }


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def generate_point_cloud(
    sampler_name: str,
    n_points: int,
    seed: int,
    samplers: Dict[str, Sampler],
) -> np.ndarray:
    """Generate a point cloud using one of the predefined samplers."""
    try:
        sampler = samplers[sampler_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sampler '{sampler_name}'. "
            f"Available: {sorted(samplers.keys())}"
        ) from exc
    return sampler(n_points, seed)


def benchmark_single_run(
    sampler_name: str,
    n_points: int,
    seed: int,
    samplers: Dict[str, Sampler],
    reduce: bool = True,
    compute_barcode: bool = True,
) -> dict:
    """Benchmark a single construction of PersistenceForest.

    Returns a dict with timing information suitable for CSV output.
    """
    # Point cloud generation
    point_cloud = generate_point_cloud(sampler_name, n_points, seed, samplers)

    # PersistenceForest construction
    t_pf_start = time.perf_counter()
    forest = PersistenceForest(
        point_cloud,
        compute=True,
        reduce=reduce,
        compute_barcode=compute_barcode,
        print_info=False,
    )
    t_pf_end = time.perf_counter()

    dim = int(point_cloud.shape[1]) if forest.point_cloud.ndim == 2 else 1

    return {
        "sampler": sampler_name,
        "n_points": int(n_points),
        "run_index": 0,  # will be overwritten by caller
        "dim": dim,
        "reduce": int(bool(reduce)),
        "compute_barcode": int(bool(compute_barcode)),
        "time_persistence_forest_s": t_pf_end - t_pf_start,
        "seed": int(seed),
    }


def benchmark_suite(
    samplers: Dict[str, Sampler],
    methods: Iterable[str],
    sizes: Iterable[int],
    n_repeats: int,
    base_seed: int,
    csv_path: str,
    reduce: bool = True,
    compute_barcode: bool = True,
) -> None:
    """Run the benchmark and write results to a CSV file.

    Parameters
    ----------
    samplers:
        Dict mapping method name -> sampler callable.
    methods:
        Iterable of keys from ``samplers`` to benchmark.
    sizes:
        Iterable of point counts.
    n_repeats:
        Number of independent runs per (method, size) pair.
    base_seed:
        Base random seed; a different seed is derived for each run but
        is deterministic and reproducible.
    csv_path:
        Output path for the CSV file with all timing data.
    reduce, compute_barcode:
        Passed through to ``PersistenceForest``. You may want to set
        ``compute_barcode=False`` if you only care about the forest
        construction time.
    """
    methods = list(methods)
    sizes = [int(s) for s in sizes]

    fieldnames = [
        "sampler",
        "n_points",
        "run_index",
        "dim",
        "reduce",
        "compute_barcode",
        "time_persistence_forest_s",
        "seed",
    ]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method in methods:
            if method not in samplers:
                raise ValueError(
                    f"Unknown sampling method '{method}'. "
                    f"Available methods: {sorted(samplers.keys())}"
                )

        for method in methods:
            print(f"starting method {method}")
            for size in sizes:
                print(f"starting size {size}")
                for r in range(n_repeats):
                    # Derive a reproducible but distinct seed for each run
                    # (simple deterministic mixing of indexes)
                    seed = base_seed + 100000 * r + 1000 * size + hash(method) % 9973

                    result = benchmark_single_run(
                        sampler_name=method,
                        n_points=size,
                        seed=seed,
                        samplers=samplers,
                        reduce=reduce,
                        compute_barcode=compute_barcode,
                    )
                    # Logical run index (0..n_repeats-1)
                    result["run_index"] = r

                    writer.writerow(result)

def plot_runtimes_from_csv(
    csv_path: str,
    methods: Sequence[str],
    time_column: str = "time_persistence_forest_s",
    ax = None,
    show_std: bool = True,
    save_dir: Optional[str]=None,
    label_dict: Optional[dict[str,str]] = None
) :
    """
    Plot runtimes vs. number of points for a list of methods.

    Parameters
    ----------
    csv_path:
        Path to the CSV produced by `benchmark_suite`.
    methods:
        Iterable of sampler names (the `sampler` column in the CSV) to plot.
    time_column:
        Which time column to use, e.g. "time_persistence_forest_s"
        or "time_total_s".
    ax:
        Optional existing matplotlib Axes to draw on. If None, a new
        figure and axes are created.
    show_std:
        If True, draw vertical error bars showing the standard deviation
        over repeated runs for each (method, n_points) pair.

    Returns
    -------
    ax:
        The matplotlib Axes containing the plot.
    """
    # Read CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data found in CSV {csv_path!r}")

    # Group times by (sampler, n_points)
    data: Dict[str, Dict[int, list[float]]] = {}
    for row in rows:
        sampler = row["sampler"]
        if sampler not in methods:
            continue

        try:
            n_points = int(row["n_points"])
        except (KeyError, ValueError):
            raise ValueError("CSV must contain an integer 'n_points' column")

        try:
            t = float(row[time_column])
        except KeyError:
            raise ValueError(
                f"CSV does not contain the requested time column {time_column!r}"
            )

        data.setdefault(sampler, {}).setdefault(n_points, []).append(t)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot each method
    for method in methods:
        if method not in data:
            print(f"[plot_runtimes_from_csv] Warning: no data for method {method!r}")
            continue

        if label_dict is not None:
            label = label_dict[method]
        else:
            label = method

        n_to_times = data[method]
        xs = sorted(n_to_times.keys())
        ys_mean = [statistics.mean(n_to_times[n]) for n in xs]

        if show_std:
            ys_std = [
                statistics.pstdev(n_to_times[n]) if len(n_to_times[n]) > 1 else 0.0
                for n in xs
            ]
            ax.errorbar(
                xs,
                ys_mean,
                yerr=ys_std,
                marker="o",
                capsize=3,
                label=label,
            )
        else:
            ax.plot(xs, ys_mean, marker="o", label=label)

    ax.set_xlabel("Number of points")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"PersistenceForest runtime vs. number of points")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_dir is not None:
        plt.savefig(save_dir, dpi =300)

    return ax

if __name__ == "__main__":

    samplers = _make_samplers()

    methods = ["uniform_2D","noisy_circle_noise-std-dot05","uniform_2D_with_30holes_radius-max-dot05"]
    methods_3d = ["uniform_3D","noisy_2sphere_noise-std-dot05","uniform_3D_with_30holes_radius-max-dot05"]

    label_dict = {"uniform_2D":"uniform 2D",
                  "noisy_circle_noise-std-dot05":"perturbed 1-sphere",
                  "uniform_2D_with_30holes_radius-max-dot05":"uniform 2D with 30 holes", 
                  "uniform_3D":"uniform 3D",
                  "noisy_2sphere_noise-std-dot05":"perturbed 2-sphere",
                  "uniform_3D_with_30holes_radius-max-dot05":"uniform 3D with 30holes"}

    name = "benchmarks/persistence_forest_benchmark_tmp"

    filename = name +".csv"
    save_dir = name +".png"

    if False:
        benchmark_suite(
            samplers=samplers,
            methods=methods + methods_3d,
            sizes=[100,250,500,750,1000, 2500,5000,7500,10000,1500,20000,30000,40000],
            n_repeats=10,
            base_seed=12345,
            csv_path=filename,
            reduce=True,
            compute_barcode=True, 
        )

    if True:
        plot_runtimes_from_csv(
        csv_path=filename,
        methods=methods_3d,
        time_column="time_persistence_forest_s",
        save_dir=save_dir,
        label_dict=label_dict
        )
        plt.show()