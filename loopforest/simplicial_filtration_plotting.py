"""
Simplicial filtration plotting helpers shared by forest classes.

The public entry point is `_plot_at_filtration_generic`, which dispatches to
2D or 3D plotting based on `forest.dim`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _camera_from_eye(camera_eye: Optional[Any]) -> tuple[float, float]:
    """
    Parse camera specification into matplotlib `(elev, azim)`.
    """
    elev, azim = 22.0, -55.0
    if camera_eye is None:
        return elev, azim
    if isinstance(camera_eye, dict):
        return float(camera_eye.get("elev", elev)), float(camera_eye.get("azim", azim))
    if isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
        return float(camera_eye[0]), float(camera_eye[1])
    raise ValueError(
        "camera_eye must be None, a dict with keys {'elev','azim'}, "
        "or a tuple/list (elev, azim)."
    )


def _plot_at_filtration_generic(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    fill_triangles: bool = True,
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_orientation_arrows: bool = False,
    remove_double_edges: bool = False,
    show_cycles: bool = True,
    linewidth_filt: float = 0.6,
    linewidth_cycle: float = 1.8,
    alpha_digits=None,
    show_complex: Optional[bool] = None,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    signed: Optional[bool] = None,
    camera_eye: Optional[Any] = None,
):
    """
    Plot simplicial filtration at a fixed filtration value.

    Dispatches to a 2D or 3D renderer based on `forest.dim`.
    """
    if forest.dim == 2:
        return _plot_at_filtration_2d(
            forest=forest,
            filt_val=filt_val,
            ax=ax,
            show=show,
            fill_triangles=fill_triangles,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            show_orientation_arrows=show_orientation_arrows,
            remove_double_edges=remove_double_edges,
            show_cycles=show_cycles,
            linewidth_filt=linewidth_filt,
            linewidth_cycle=linewidth_cycle,
            alpha_digits=alpha_digits,
        )
    if forest.dim == 3:
        if show_complex is None:
            show_complex = fill_triangles
        if signed is None:
            signed = not remove_double_edges
        return _plot_at_filtration_3d(
            forest=forest,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            show_cycles=show_cycles,
            signed=signed,
            min_bar_length=0.0,
            complex_opacity=complex_opacity,
            cycle_opacity=cycle_opacity,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            linewidth_filt=linewidth_filt,
            linewidth_cycle=linewidth_cycle,
            camera_eye=camera_eye,
        )

    raise ValueError("plot_at_filtration is only implemented for ambient dimensions 2 and 3.")


def _plot_at_filtration_2d(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    fill_triangles: bool = True,
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_orientation_arrows: bool = False,
    remove_double_edges: bool = False,
    show_cycles: bool = True,
    linewidth_filt: float = 0.6,
    linewidth_cycle: float = 1.8,
    alpha_digits=None,
):
    """
    Plot 2D filtration; behavior is kept compatible with prior implementation.
    """
    color_map = forest._get_color_map(coloring=coloring)

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("point_cloud must be an (n_points, 2) array-like.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    edges_xy = []
    tris_xy = []
    for simplex, f in forest.filtration:
        if f > filt_val:
            break
        if len(simplex) == 2:
            i, j = simplex
            edges_xy.append([pts[i], pts[j]])
        elif len(simplex) == 3:
            i, j, k = simplex
            tris_xy.append([pts[i], pts[j], pts[k]])

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color="k",
        label="points",
        marker="o",
        edgecolors="none",
        zorder=2.8,
    )

    if fill_triangles and tris_xy:
        tri_coll = PolyCollection(
            tris_xy,
            closed=True,
            edgecolors="none",
            facecolors="C0",
            alpha=0.2,
            zorder=1,
        )
        ax.add_collection(tri_coll)

    if edges_xy:
        edge_coll = LineCollection(
            edges_xy,
            linewidths=linewidth_filt,
            colors="0.3",
            zorder=2,
            label="edges",
        )
        ax.add_collection(edge_coll)

    if show_cycles:
        for bar in forest.barcode:
            if filt_val >= bar.birth and filt_val < bar.death:
                cycle = bar.cycle_at_filtration_value(filt_val=filt_val)
                segments = forest._chain_segments_2d(
                    chain=cycle,
                    signed=(not remove_double_edges),
                )

                loop_coll = LineCollection(
                    segments,
                    linewidths=linewidth_cycle,
                    colors=[color_map[bar]],
                    zorder=5,
                )
                ax.add_collection(loop_coll)

                if show_orientation_arrows:
                    for seg in segments:
                        (x0, y0), (x1, y1) = np.asarray(seg, dtype=float)
                        dx = x1 - x0
                        dy = y1 - y0
                        length = float(np.hypot(dx, dy))
                        if length == 0.0:
                            continue

                        frac = 0.5
                        mx = 0.5 * (x0 + x1)
                        my = 0.5 * (y0 + y1)
                        ux = dx / length
                        uy = dy / length

                        half = 0.5 * frac * length
                        x_start = mx - ux * half
                        y_start = my - uy * half
                        x_end = mx + ux * half
                        y_end = my + uy * half

                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x_start, y_start),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                linewidth=0.2,
                                color=color_map[bar],
                                mutation_scale=6,
                            ),
                            zorder=6,
                        )

    ax.set_aspect("equal", adjustable="box")
    if title is None:
        ax.set_title(fr"Filtration at radius r= {filt_val:.4g} ")
    else:
        ax.set_title(title)

    ax.autoscale()
    if show:
        plt.show()
    return ax


def _plot_at_filtration_3d(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    show_complex: bool = True,
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3.0,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    linewidth_filt: float = 0.6,
    linewidth_cycle: float = 0.2,
    camera_eye: Optional[Any] = None,
):
    """
    Plot 3D filtration snapshot with optional complex boundary and cycle surfaces.
    """
    color_map = forest._get_color_map(coloring=coloring)
    snapshot = forest._complex_snapshot_at_filtration(filt_val=float(filt_val))
    pts = np.asarray(snapshot["points"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")

    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        created_ax = True
    elif not hasattr(ax, "zaxis"):
        raise ValueError("3D plotting requires a matplotlib 3D axis (projection='3d').")

    if show_complex:
        edges = snapshot.get("edges", [])
        if edges:
            segments = [pts[list(edge)] for edge in edges]
            edge_coll = Line3DCollection(
                segments,
                colors="0.35",
                linewidths=linewidth_filt,
                alpha=max(0.2, float(complex_opacity)),
            )
            ax.add_collection3d(edge_coll)

        triangles = snapshot.get("triangles", [])
        if triangles:
            tri_polys = [pts[list(tri)] for tri in triangles]
            tri_coll = Poly3DCollection(
                tri_polys,
                facecolors=(0.678, 0.847, 0.902, float(complex_opacity)),  # lightblue RGBA
                edgecolors="none",
            )
            ax.add_collection3d(tri_coll)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=vertex_size,
        c="black",
        depthshade=False,
    )

    if show_cycles:
        active = forest._active_bars_with_cycles_at(
            filt_val=float(filt_val),
            min_bar_length=min_bar_length,
        )
        active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

        from matplotlib import colors as mcolors
        for bar, cycle in active:
            tri_faces = forest._chain_triangles_3d(cycle, signed=signed)
            if not tri_faces:
                continue
            cycle_polys = [pts[list(face)] for face in tri_faces]
            color = color_map.get(bar, "#d62728")
            cycle_coll = Poly3DCollection(
                cycle_polys,
                facecolors=mcolors.to_rgba(color, alpha=cycle_opacity),
                edgecolors=mcolors.to_rgba(color, alpha=min(1.0, cycle_opacity + 0.25)),
                linewidths=linewidth_cycle,
            )
            ax.add_collection3d(cycle_coll)

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    pad = 0.05 * np.max(spans)
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

    elev, azim = _camera_from_eye(camera_eye=camera_eye)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is None:
        ax.set_title(fr"Filtration at radius r= {filt_val:.4g} ")
    else:
        ax.set_title(title)

    if show and created_ax:
        plt.show()
    elif show:
        plt.show()

    return ax
