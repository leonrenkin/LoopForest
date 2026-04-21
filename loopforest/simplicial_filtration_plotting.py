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


def _desaturate_color(color: Any, amount: float) -> tuple[float, float, float]:
    """
    Return an RGB color with saturation reduced by `amount` in [0, 1].
    """
    from matplotlib import colors as mcolors

    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] *= 1.0 - amount
    out = mcolors.hsv_to_rgb(hsv)
    return (float(out[0]), float(out[1]), float(out[2]))


def _resolve_style_2d(style_2d: Optional[dict[str, Any]]) -> dict[str, Any]:
    style = {
        "point_color": "k",
        "point_alpha": 1.0,
        "complex_face_color": "C0",
        "complex_face_alpha": 0.2,
        "complex_edge_color": "0.3",
        "complex_edge_width": 0.6,
        "cycle_edge_width": 1.8,
        "show_orientation_arrows": False,
        "arrow_linewidth": 0.8,
        "arrow_scale": 12.0,
    }
    if style_2d:
        style.update(style_2d)
    return style


def _resolve_style_3d(style_3d: Optional[dict[str, Any]]) -> dict[str, Any]:
    style = {
        "camera_eye": None,
        "remove_axes": False,
        "point_color": "black",
        "point_alpha": 1.0,
        "depthshade_points": False,
        "complex_color": "#add8e6",
        "complex_face_alpha": 0.20,
        "cycle_face_alpha": 0.55,
        "complex_edge_color": "0.35",
        "cycle_edge_color": None,
        "complex_edge_width": 0.6,
        "cycle_edge_width": 0.2,
        "complex_edge_alpha": None,
        "cycle_edge_alpha": None,
        "antialiased": True,
        "zsort": "average",
        "desaturate_complex": 0.0,
    }
    if style_3d:
        style.update(style_3d)
    return style


def _plot_at_filtration_generic(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    show_complex: bool = True,
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_cycles: bool = True,
    signed: bool = False,
    style_2d: Optional[dict[str, Any]] = None,
    style_3d: Optional[dict[str, Any]] = None,
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
            show_complex=show_complex,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            show_cycles=show_cycles,
            signed=signed,
            style_2d=style_2d,
        )
    if forest.dim == 3:
        return _plot_at_filtration_3d(
            forest=forest,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            show_cycles=show_cycles,
            signed=signed,
            min_bar_length=0.0,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            style_3d=style_3d,
        )

    raise ValueError("plot_at_filtration is only implemented for ambient dimensions 2 and 3.")


def _plot_at_filtration_2d(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    show_complex: bool = True,
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_cycles: bool = True,
    signed: bool = False,
    style_2d: Optional[dict[str, Any]] = None,
):
    """
    Plot 2D filtration; behavior is kept compatible with prior implementation.
    """
    color_map = forest._get_color_map(coloring=coloring)
    style = _resolve_style_2d(style_2d)

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
        color=style["point_color"],
        alpha=float(style["point_alpha"]),
        label="points",
        marker="o",
        edgecolors="none",
        zorder=2.8,
    )

    if show_complex and tris_xy:
        tri_coll = PolyCollection(
            tris_xy,
            closed=True,
            edgecolors="none",
            facecolors=style["complex_face_color"],
            alpha=float(style["complex_face_alpha"]),
            zorder=1,
        )
        ax.add_collection(tri_coll)

    if show_complex and edges_xy:
        edge_coll = LineCollection(
            edges_xy,
            linewidths=float(style["complex_edge_width"]),
            colors=style["complex_edge_color"],
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
                    signed=signed,
                )

                loop_coll = LineCollection(
                    segments,
                    linewidths=float(style["cycle_edge_width"]),
                    colors=[color_map[bar]],
                    zorder=5,
                )
                ax.add_collection(loop_coll)

                if bool(style["show_orientation_arrows"]):
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
                                linewidth=float(style["arrow_linewidth"]),
                                color=color_map[bar],
                                mutation_scale=float(style["arrow_scale"]),
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
    figsize: tuple[float, float] = (7, 7),
    vertex_size: float = 3.0,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    style_3d: Optional[dict[str, Any]] = None,
):
    """
    Plot 3D filtration snapshot with optional complex boundary and cycle surfaces.
    """
    color_map = forest._get_color_map(coloring=coloring)
    style = _resolve_style_3d(style_3d)
    complex_face_alpha = float(style["complex_face_alpha"])
    cycle_face_alpha = float(style["cycle_face_alpha"])
    complex_edge_alpha = style["complex_edge_alpha"]
    if complex_edge_alpha is None:
        complex_edge_alpha = max(0.2, complex_face_alpha)
    cycle_edge_alpha = style["cycle_edge_alpha"]
    if cycle_edge_alpha is None:
        cycle_edge_alpha = min(1.0, cycle_face_alpha + 0.25)
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
                colors=style["complex_edge_color"],
                linewidths=float(style["complex_edge_width"]),
                alpha=float(complex_edge_alpha),
                antialiased=bool(style["antialiased"]),
            )
            ax.add_collection3d(edge_coll)

        triangles = snapshot.get("triangles", [])
        if triangles:
            tri_polys = [pts[list(tri)] for tri in triangles]
            complex_rgb = _desaturate_color(style["complex_color"], float(style["desaturate_complex"]))
            tri_coll = Poly3DCollection(
                tri_polys,
                facecolors=(*complex_rgb, complex_face_alpha),
                edgecolors="none",
                antialiased=bool(style["antialiased"]),
                zsort=style["zsort"],
            )
            ax.add_collection3d(tri_coll)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=vertex_size,
        c=style["point_color"],
        alpha=float(style["point_alpha"]),
        depthshade=bool(style["depthshade_points"]),
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
            cycle_edge_color_eff = (
                color if style["cycle_edge_color"] is None else style["cycle_edge_color"]
            )
            cycle_coll = Poly3DCollection(
                cycle_polys,
                facecolors=mcolors.to_rgba(color, alpha=cycle_face_alpha),
                edgecolors=mcolors.to_rgba(cycle_edge_color_eff, alpha=float(cycle_edge_alpha)),
                linewidths=float(style["cycle_edge_width"]),
                antialiased=bool(style["antialiased"]),
                zsort=style["zsort"],
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

    elev, azim = _camera_from_eye(camera_eye=style["camera_eye"])
    ax.view_init(elev=elev, azim=azim)
    if bool(style["remove_axes"]):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
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
