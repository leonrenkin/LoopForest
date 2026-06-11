from __future__ import annotations

from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection


def plot_interior_simplex_activity(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    overlap: Literal["longest", "layer"] = "longest",
    vertex_size: float = 2,
    title: str | None = None,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
):
    """
    Plot 2D filtration triangles colored by interior simplex activity.

    The input forest must already have interior cycle representatives available.
    Activity data is read from ``forest.interior_simplex_activity()``.
    """
    if forest.dim != 2:
        raise ValueError("plot_interior_simplex_activity only supports 2D PersistenceForest objects.")
    if overlap not in ("longest", "layer"):
        raise ValueError("overlap must be 'longest' or 'layer'.")

    plot_style = {
        "point_color": "black",
        "point_alpha": 0.9,
        "activity_edge_color": "white",
        "activity_edge_width": 0.25,
        "complex_edge_color": "0",
        "complex_edge_width": 0.45,
        "complex_edge_alpha": 0.85,
        "background_color": "white",
        "remove_axes": True,
        "activity_alpha_range": (0.15, 0.95),
    }
    if style is not None:
        plot_style.update(style)

    pts = np.asarray(forest.point_cloud, dtype=float)
    color_map = forest._get_color_map(coloring=coloring)
    activity = forest.interior_simplex_activity()

    rows = []
    for simplex_key, simplex_activity in activity.items():
        for bar, active_start, active_end in simplex_activity:
            activity_length = float(active_end - active_start)
            if activity_length >= min_activity_length:
                rows.append((tuple(simplex_key), bar, activity_length))

    if overlap == "longest":
        longest_by_simplex = {}
        for simplex_key, bar, activity_length in rows:
            current = longest_by_simplex.get(simplex_key)
            if current is None or activity_length > current[1]:
                longest_by_simplex[simplex_key] = (bar, activity_length)
        rows = [
            (simplex_key, bar, activity_length)
            for simplex_key, (bar, activity_length) in longest_by_simplex.items()
        ]
    else:
        rows.sort(key=lambda row: row[2])

    max_activity_length = max((activity_length for _, _, activity_length in rows), default=0.0)
    alpha_min, alpha_max = plot_style["activity_alpha_range"]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.set_facecolor(plot_style["background_color"])

    if rows:
        triangle_polys = []
        triangle_colors = []

        for simplex_key, bar, activity_length in rows:
            alpha_scale = activity_length / max_activity_length
            alpha = float(alpha_min + (alpha_max - alpha_min) * alpha_scale)
            triangle_polys.append(pts[list(simplex_key)])
            triangle_colors.append(mcolors.to_rgba(color_map[bar], alpha=alpha))

        activity_collection = PolyCollection(
            triangle_polys,
            closed=True,
            facecolors=triangle_colors,
            edgecolors=plot_style["activity_edge_color"],
            linewidths=float(plot_style["activity_edge_width"]),
            zorder=1,
        )
        ax.add_collection(activity_collection)

    if show_complex:
        edge_segments = []
        for simplex, _filtration in forest.filtration:
            if len(simplex) == 2:
                edge_segments.append(pts[list(simplex)])

        if edge_segments:
            edge_collection = LineCollection(
                edge_segments,
                colors=plot_style["complex_edge_color"],
                linewidths=float(plot_style["complex_edge_width"]),
                alpha=float(plot_style["complex_edge_alpha"]),
                zorder=2,
            )
            ax.add_collection(edge_collection)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color=plot_style["point_color"],
        alpha=float(plot_style["point_alpha"]),
        edgecolors="none",
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

    if bool(plot_style["remove_axes"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show:
        plt.show()

    return ax
