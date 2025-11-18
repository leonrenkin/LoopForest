"""
Shared plotting utilities for LoopForest and PersistenceForest.

This module is designed to be *forest-agnostic*: it only assumes that a
"forest" object provides

    - forest.barcode        : iterable of bar objects
    - each bar has .birth, .death, and preferably .lifespan()
    - (optionally) forest._build_color_map_forest()
                    forest._build_color_map_bars()
                    forest.color_map_forest
                    forest.color_map_bars
    - (for animations) forest.filtration : iterable of (simplex, filt_val)
      and a method
          forest.plot_at_filtration(filt_val: float, ax=None, **kwargs)

Both `LoopForest` and `PersistenceForest` in this project satisfy these
assumptions, so they can re-use these utilities.

Typical use inside a class:

    from forest_plotting import plot_barcode as _plot_barcode_generic
    from forest_plotting import animate_filtration as _animate_filtration_generic

    class PersistenceForest:
        ...
        def plot_barcode(self, *args, **kwargs):
            return _plot_barcode_generic(self, *args, **kwargs)

        def animate_filtration(self, *args, **kwargs):
            return _animate_filtration_generic(self, *args, **kwargs)

You are free to adapt the wrappers (defaults, docstrings, etc.) per class.
"""

from typing import Literal, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _plot_barcode_generic(
        forest,
        *,
        ax=None,
        sort: str | None = "birth",   # "length" | "birth" | "death" | None
        title: str = "Barcode",
        xlabel: str = "filtration value",
        coloring: Literal["forest", "bars"] = "forest",
        max_bars: int = 0,
        min_bar_length: float = 0.0,
    ):
    """
    Plot a 1D barcode from forest.barcode (a set[Bar]).

    Each Bar contributes a horizontal segment from birth to death.
    If death is +inf, an arrow is drawn to the right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        If given, draw on this axes. Otherwise a new figure/axes is created.
    sort : {"length","birth","death",None}
        Sort bars before plotting (None preserves current order).
        Default is "birth".
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    coloring : {"forest","bars"}
        Which color scheme to use:
        - "forest": use forest.color_map_forest (tree-structured colors).
        - "bars":   use forest.color_map_bars (ignores tree structure).
        If the chosen color map does not exist yet, it is built as in
        `plot_at_filtration`.
    max_bars : int
        If > 0, display at most this many bars, keeping the longest ones
        (by lifespan). 0 means show all bars.
    min_bar_length : float
        Filter out bars with lifespan < min_bar_length before plotting.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the barcode was drawn on.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if not getattr(forest, "barcode", None):
        raise ValueError("No bars to plot: `forest.barcode` is empty.")

    # ---- Prepare color map (same logic as plot_at_filtration) ----
    if coloring == "forest":
        if not hasattr(forest, "color_map_forest"):
            forest._build_color_map_forest()
        color_map = forest.color_map_forest
    elif coloring == "bars":
        if not hasattr(forest, "color_map_bars"):
            forest._build_color_map_bars()
        color_map = forest.color_map_bars
    else:
        # Fallback: no special coloring
        color_map = {}

    # ---- Work on a copy so we don't mutate original order ----
    bars = list(forest.barcode)

    # Filter by minimum length (Gudhi-like)
    if min_bar_length > 0.0:
        bars = [b for b in bars if b.lifespan() >= min_bar_length]

    if not bars:
        raise ValueError(
            "No bars to plot after applying min_bar_length filter "
            f"(min_bar_length = {min_bar_length})."
        )

    # Limit to longest `max_bars` bars if requested (Gudhi-like)
    if max_bars and max_bars > 0 and len(bars) > max_bars:
        bars = sorted(bars, key=lambda b: b.lifespan(), reverse=True)[:max_bars]

    # Optional sorting for display
    if sort == "birth":
        bars.sort(key=lambda b: (b.birth, b.death))
    elif sort == "death":
        def dkey(b):
            d = b.death
            return (math.inf if not math.isfinite(d) else d, b.birth)
        bars.sort(key=dkey)
    elif sort == "length":
        def length(b):
            d = b.death
            d_val = math.inf if not math.isfinite(d) else d
            return d_val - b.birth
        bars.sort(key=length, reverse=True)
    elif sort is None:
        # Keep whatever order came out of filtering
        pass
    else:
        raise ValueError(
            f"Unknown sort option {sort!r}. "
            "Expected one of 'birth', 'death', 'length', or None."
        )

    n_bars = len(bars)

    # ---- Create axes if needed, with controlled figure height ----
    created_ax = False
    if ax is None:
        # Height grows sublinearly and is capped to avoid gigantic figures
        base_height = 2.5
        extra_height = 0.12 * min(n_bars, 80)   # at most ~9.1 total
        fig, ax = plt.subplots(figsize=(7, base_height + extra_height))
        created_ax = True
    else:
        fig = ax.figure

    # ---- Determine x-limits with a bit of padding ----
    births = np.array([b.birth for b in bars], dtype=float)
    deaths = np.array([b.death for b in bars], dtype=float)
    finite_deaths = deaths[np.isfinite(deaths)]

    xmin = float(np.nanmin(births))
    if finite_deaths.size:
        xmax = float(np.nanmax(finite_deaths))
    else:
        xmax = float(np.nanmax(births))

    if not np.isfinite(xmax):  # extreme corner case
        xmax = xmin

    pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    ax.set_xlim(xmin - pad, xmax + pad)

    # ---- Draw segments ----
    for i, b in enumerate(bars):
        x0, x1 = float(b.birth), float(b.death)

        # Guard against inverted bars due to numerical issues
        if math.isfinite(x1) and x1 < x0:
            x0, x1 = x1, x0

        color = color_map.get(b, None)
        line_kwargs = {
            "y": i,
            "xmin": x0,
            "xmax": x1 if math.isfinite(x1) else ax.get_xlim()[1] - 0.25 * pad,
            "linewidth": 3.0,  # thicker bars
        }
        if color is not None:
            line_kwargs["color"] = color

        if math.isfinite(x1):
            # Finite bar: simple thick line, NO endpoint markers
            ax.hlines(**line_kwargs)
        else:
            # Infinite bar: truncated line + arrow
            right = ax.get_xlim()[1]
            line_kwargs["xmax"] = right - 0.25 * pad
            ax.hlines(**line_kwargs)

            # Draw arrow for infinity
            arrow_kwargs = {
                "xy":   (right - 0.15 * pad, i),
                "xytext": (x0, i),
                "arrowprops": dict(arrowstyle="->", lw=2),
                "va": "center",
            }
            if color is not None:
                arrow_kwargs["arrowprops"]["color"] = color

            ax.annotate("", **arrow_kwargs)

    # ---- Cosmetics ----
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-1, n_bars)  # keep bars nicely framed
    fig.tight_layout()

    # If we created the axes, show it immediately (so this works in scripts)
    if created_ax:
        import matplotlib.pyplot as plt
        plt.show()

    return ax

def _plot_dendrogram_generic(
        forest,
        ax=None,
        show: bool = True,
        annotate_ids: bool = False,
        leaf_spacing: float = 1.0,
        tree_gap_leaves: int = 1,
        check_reduced: bool = True,
        small_on_top: bool = False,
        threshold: float = 0.0
    ):
    import warnings

    if not forest.nodes:
        raise ValueError("LoopForest has no nodes to plot.")

    all_nodes = forest.nodes
    all_ids = set(all_nodes.keys())

    # Recompute roots robustly from the current structure (these are Node objects)
    all_roots = [n for n in all_nodes.values() if n.parent is None or n.parent not in all_ids]

    if check_reduced:
        equal_pairs = [
            (p.id, c)
            for p in all_nodes.values()
            for c in p.children
            if c in all_nodes and all_nodes[c].filt_val == p.filt_val
        ]
        if equal_pairs:
            warnings.warn(
                f"Forest does not appear reduced: found {len(equal_pairs)} parent–child pairs "
                f"with equal filt_val. Plotting anyway."
            )

    bad_direction = [
        (p.id, c)
        for p in all_nodes.values()
        for c in p.children
        if c in all_nodes and all_nodes[c].filt_val < p.filt_val
    ]
    if bad_direction:
        warnings.warn(
            f"{len(bad_direction)} edges have child.filt_val > parent.filt_val."
        )

    # ---------- threshold filtering (key addition) ----------
    def _subtree_ids(root_id: int) -> set[int]:
        """All node ids reachable from (and including) root_id that are present in all_nodes."""
        stack = [root_id]
        seen: set[int] = set()
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in all_nodes:
                continue
            seen.add(nid)
            stack.extend([cid for cid in all_nodes[nid].children if cid in all_nodes])
        return seen

    def _is_leaf_in(sub_ids: set[int], nid: int) -> bool:
        """Leaf = no children inside this same subgraph."""
        return not any((cid in sub_ids) for cid in all_nodes[nid].children)

    included_root_ids: list[int] = []
    included_ids: set[int] = set()

    for r in all_roots:
        sub_ids = _subtree_ids(r.id)
        # Identify leaves within this subgraph
        leaves = [nid for nid in sub_ids if _is_leaf_in(sub_ids, nid)]
        root_val = float(all_nodes[r.id].filt_val)
        max_leaf_delta = max((abs(float(all_nodes[l].filt_val) - root_val) for l in leaves), default=0.0)

        if max_leaf_delta > float(threshold):
            included_root_ids.append(r.id)
            included_ids.update(sub_ids)

    if threshold <= 0.0:
        # No filtering requested: include everything
        nodes = all_nodes
        roots = sorted(all_roots, key=lambda n: (n.filt_val, n.id))
    else:
        if not included_ids:
            # Nothing to plot under this threshold — return an empty/annotated axes.
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"LoopForest dendrogram (y = filt_val) — no trees exceed threshold {threshold}")
            ax.set_axis_off()
            if show:
                plt.show()
            return ax
        nodes = {nid: all_nodes[nid] for nid in included_ids}
        roots = [all_nodes[rid] for rid in included_root_ids]
        roots.sort(key=lambda n: (n.filt_val, n.id))

    node_ids = set(nodes.keys())

    # ---------- positions ----------
    x: dict[int, float] = {}
    y: dict[int, float] = {n.id: float(n.filt_val) for n in nodes.values()}
    visited: set[int] = set()
    leaf_counter = 0

    def _assign_x(nid: int):
        nonlocal leaf_counter
        if nid in visited:
            return
        visited.add(nid)

        child_ids = [cid for cid in nodes[nid].children if cid in nodes]
        child_ids.sort(key=lambda cid: (nodes[cid].filt_val, nodes[cid].id))

        if len(child_ids) == 0:
            x[nid] = leaf_counter * leaf_spacing
            leaf_counter += 1
        else:
            for cid in child_ids:
                _assign_x(cid)
            xs = [x[cid] for cid in child_ids]
            x[nid] = sum(xs) / len(xs)

    # Lay out each tree; insert spacing between trees
    for i, r in enumerate(roots):
        start_before = leaf_counter
        _assign_x(r.id)
        if i != len(roots) - 1 and leaf_counter > start_before:
            leaf_counter += tree_gap_leaves

    # Place any stray components (shouldn't happen, but be safe) — within the filtered set only
    for nid in list(node_ids):
        if nid not in x:
            _assign_x(nid)
            leaf_counter += tree_gap_leaves

    # ---------- draw ----------
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    line_color = "0.25"
    merge_color = "0.1"
    node_edge = "white"

    for p in nodes.values():
        pid = p.id
        px, py = x[pid], y[pid]
        child_ids = [cid for cid in p.children if cid in nodes]
        if not child_ids:
            continue

        child_ids.sort(key=lambda cid: x[cid])
        child_xs = [x[cid] for cid in child_ids]
        child_ys = [y[cid] for cid in child_ids]

        for cx, cy in zip(child_xs, child_ys):
            ax.plot([cx, cx], [cy, py], linewidth=1.2, color=line_color, zorder=1)

        if len(child_ids) >= 2:
            ax.plot([min(child_xs), max(child_xs)], [py, py], linewidth=1.5, color=merge_color, zorder=2)

    for n in nodes.values():
        ax.scatter(x[n.id], y[n.id], s=24, zorder=3, color="C0", edgecolors=node_edge, linewidths=0.6)
        if annotate_ids:
            ax.annotate(
                str(n.id),
                (x[n.id], y[n.id]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="0.2",
            )

    ax.set_xlabel("leaf order")
    ax.set_ylabel("filt_val (y)")
    title = "LoopForest dendrogram (y = filt_val)"
    if threshold > 0.0:
        title += f" — threshold > {threshold}"
    ax.set_title(title)
    ax.margins(x=0.05, y=0.05)
    ax.grid(False)

    if x:
        xs = list(x.values())
        ax.set_xlim(min(xs) - leaf_spacing, max(xs) + leaf_spacing)

    if small_on_top:
        ax.invert_yaxis()

    if show:
        plt.show()
    return ax

def _animate_filtration_generic(
        forest,
        filename: Optional[str] = None,
        *,
        fps: int = 20,
        frames: int = 200,
        coloring: Literal["forest", "bars"] = "forest",
        with_barcode: bool = False,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 200,
        cloud_figsize: tuple[float, float] = (6.0, 6.0),
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
    ):
        """
        Create an animation of the loop forest over the filtration.

        Parameters
        ----------
        filename : str | None, optional
            If given, the animation is written to this path.
        fps : int, optional
            Frames per second for the saved animation.
        frames : int, optional
            Number of time steps (frames) sampled between t_min and t_max.
        with_barcode : bool, optional
            If True, show a second panel with the barcode and a moving vertical
            line indicating the current filtration value.
        t_min, t_max : float | None, optional
            Optional lower/upper bounds on the filtration values to animate.
        dpi : int, optional
            DPI for saving the animation.
        cloud_figsize : (float, float), optional
            Size of the point-cloud panel when with_barcode=False.
        total_figsize : (float, float) | None, optional
            Total figure size when with_barcode=True. If None, a reasonable
            default (10, 5) is used.
        plot_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``plot_at_filtration``.
            For example::
                plot_kwargs=dict(
                    fill_triangles=True,
                    loop_vertex_markers=False,
                    point_size=3,
                    coloring="forest",
                )
        barcode_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``_plot_barcode`` **except**
            ``ax`` and ``coloring``, which are managed by this method.
            For example::
                barcode_kwargs=dict(
                    max_bars=150,
                    min_bar_length=1e-3,
                    sort="length",
                    title="Barcode",
                )

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The created animation. If ``filename`` is not None, the animation
            is also saved to disk.
        fig : matplotlib.figure.Figure
            The figure on which the animation is drawn.
        """
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError("LoopForest has no filtration data to animate.")

        # Optional restriction to a time window
        if t_min is None:
            t_min = 0.0
        if t_max is None:
            t_max = max(bar.death for bar in forest.barcode)

        # Uniformly spaced in filtration value → uniform speed
        frame_times = np.linspace(t_min, t_max, frames).tolist()  # pyright: ignore[reportCallIssue, reportArgumentType]

        # ---- Common kwargs for plot_at_filtration (cloud panel) ----
        if plot_kwargs is None:
            plot_kwargs = {}
        # Reasonable defaults (only used if not explicitly overridden)
        plot_kwargs = {
            "fill_triangles": True,
            "loop_vertex_markers": False,
            "point_size": 3,
            "coloring": coloring,
            "show": False,   # important: we manage the figure ourselves
            **plot_kwargs,
        }

        # ---- Barcode kwargs & shared color dict ----
        if with_barcode:
            if total_figsize is None:
                total_figsize = (10.0, 5.0)

            fig, (ax_cloud, ax_bar) = plt.subplots(
                1, 2,
                figsize=total_figsize,
                gridspec_kw={"width_ratios": [3, 2]},
            )

            # Draw the (static) barcode once
            if not getattr(forest, "barcode", None):
                raise ValueError("`with_barcode=True` but `forest.barcode` is empty.")

            if barcode_kwargs is None:
                barcode_kwargs = {}
            # Do not let the caller override ax or coloring here
            barcode_kwargs = {
                k: v for k, v in barcode_kwargs.items()
                if k not in {"ax", "coloring"}
            }
            # Defaults for the barcode panel – user can override sort/title/xlabel
            barcode_kwargs = {
                "sort": "length",
                "title": "Barcode",
                "xlabel": "filtration value",
                **barcode_kwargs,
            }
            # Enforce *same* coloring as in plot_at_filtration
            barcode_kwargs["coloring"] = plot_kwargs[coloring]

            forest.plot_barcode(
                ax=ax_bar,
                **barcode_kwargs,
            )

            # Vertical line that will move with the filtration
            current_t0 = frame_times[0]
            barcode_line = ax_bar.axvline(current_t0, color="k", linewidth=2)
        else:
            fig, ax_cloud = plt.subplots(figsize=cloud_figsize)
            ax_bar = None
            barcode_line = None

        # ---- Helper to draw a single frame on the cloud panel ----
        def _draw_frame_at_time(t: float):
            """Helper: clear the cloud axis and redraw for filtration value t."""
            ax_cloud.clear()
            # Delegate the heavy lifting to the existing helper
            forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **plot_kwargs)

            # Optional: overlay a small text box with the current filtration value.
            # Comment this out if you prefer only the built-in title.
            ax_cloud.text(
                0.02, 0.98, rf"$\alpha = {t:.3g}$",
                transform=ax_cloud.transAxes,
                va="top", ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        # ---- Animation callbacks ----
        def init():
            _draw_frame_at_time(frame_times[0])
            if with_barcode and barcode_line is not None:
                t0 = frame_times[0]
                barcode_line.set_xdata([t0, t0])
            return []

        def update(frame_idx: int):
            t = frame_times[frame_idx]
            _draw_frame_at_time(t)
            if with_barcode and barcode_line is not None:
                barcode_line.set_xdata([t, t])
            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frame_times),
            init_func=init,
            blit=False,
        )

        # ---- Optionally write to disk ----
        if filename is not None:
            fname = str(filename)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if ext == "mp4":
                writer = FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(fname, writer=writer, dpi=dpi)
            elif ext in {"gif", "gifv"}:
                try:
                    from matplotlib.animation import PillowWriter
                except ImportError as e:  # optional dependency
                    raise RuntimeError(
                        "Saving as GIF requires Pillow. Install it with `pip install pillow`."
                    ) from e
                writer = PillowWriter(fps=fps)
                anim.save(fname, writer=writer, dpi=dpi)
            else:
                # Fallback to default writer chosen by matplotlib
                anim.save(fname, dpi=dpi, fps=fps)

        return anim, fig

def animate_filtration_pair(
    forest1,
    forest2,
    filename: Optional[str] = None,
    *,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dpi: int = 200,
    total_figsize: Optional[Tuple[float, float]] = None,
    plot_kwargs_forest1: Optional[dict] = None,
    plot_kwargs_forest2: Optional[dict] = None,
    barcode_kwargs_forest1: Optional[dict] = None,
    barcode_kwargs_forest2: Optional[dict] = None,
):
    """
    Animate two LoopForests side-by-side: for each forest, show the evolving
    cycle representatives in the point cloud together with its barcode.

    The filtration panels and barcodes are styled via `plot_at_filtration` and
    `_plot_barcode`, so they match what `animate_filtration` produces.

    Parameters
    ----------
    forest1, forest2 : LoopForest
        The two forests to animate.
    filename : str or None, optional
        If not None, the animation is also written to this file.
        The extension ('.mp4', '.gif', etc.) determines the writer.
    fps : int, optional
        Frames per second for the saved animation.
    frames : int, optional
        Number of frames in the animation (shared time grid).
    t_min, t_max : float or None, optional
        Filtration time window. If None, t_min = 0 and t_max is the max
        filtration value across both forests.
    dpi : int, optional
        DPI for saving.
    total_figsize : (float, float) or None, optional
        Overall figure size. If None, a reasonable default is used.
    plot_kwargs_forest1, plot_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `plot_at_filtration` for each forest.
        Used on the *cloud/loop* panels.
    barcode_kwargs_forest1, barcode_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `_plot_barcode` for each forest.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # --- 1) Sanity checks -----------------------------------------------------
    for forest, name in ((forest1, "forest1"), (forest2, "forest2")):
        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError(f"{name} has no filtration data to animate.")
        if not getattr(forest, "barcode", None):
            raise ValueError(f"{name} has an empty barcode.")

    # --- 2) Time grid ---------------------------------------------------------
    if t_min is None:
        t_start: float = 0.0
    else:
        t_start = float(t_min)

    if t_max is None:
        max1 = max(node.filt_val for node in forest1.nodes.values())
        max2 = max(node.filt_val for node in forest2.nodes.values())
        t_end: float = float(max(max1, max2))*1.05
    else:
        t_end = float(t_max)*1.05

    n_frames: int = int(frames)

    frame_times = np.linspace(start = t_start, stop = t_end, num = n_frames).tolist()

    # --- 3) Figure layout: 2x2 (clouds on top, barcodes below) ----------------
    if total_figsize is None:
        total_figsize = (12.0, 8.0)

    fig, ((ax_cloud_1, ax_cloud_2),
          (ax_bar_1,   ax_bar_2)) = plt.subplots(
        2,
        2,
        figsize=total_figsize,
        gridspec_kw={"height_ratios": [5, 2]},
    )

    # --- 4) Barcode plotting --------------------------------------------------
    # Base kwargs to resemble your existing barcode style
    base_barcode_kwargs = {
        "sort": "length",
        "xlabel": "filtration value",
    }

    if barcode_kwargs_forest1 is None:
        barcode_kwargs_forest1 = {}
    if barcode_kwargs_forest2 is None:
        barcode_kwargs_forest2 = {}

    kwargs_bar_1 = {**base_barcode_kwargs, **barcode_kwargs_forest1}
    kwargs_bar_2 = {**base_barcode_kwargs, **barcode_kwargs_forest2}

    if "title" not in kwargs_bar_1:
        kwargs_bar_1["title"] = "Barcode"
    if "title" not in kwargs_bar_2:
        kwargs_bar_2["title"] = "Barcode"

    forest1.plot_barcode(ax=ax_bar_1, **kwargs_bar_1)
    forest2.plot_barcode(ax=ax_bar_2, **kwargs_bar_2)

    # Force both barcodes to share the same x-range as the animation time
    ax_bar_1.set_xlim(t_start, t_end)
    ax_bar_2.set_xlim(t_start, t_end)

    # Vertical lines tracking the current filtration value
    t0 = frame_times[0]
    barcode_line_1 = ax_bar_1.axvline(t0, color="k", linewidth=2)
    barcode_line_2 = ax_bar_2.axvline(t0, color="k", linewidth=2)

    # --- 5) Filtration plot kwargs (match animate_filtration defaults) --------
    base_plot_kwargs = {
        "fill_triangles": True,
        "loop_vertex_markers": False,
        "point_size": 3,
        "coloring": "forest",  # uses the forest's color dict, shared with barcode
        "show": False,         # we manage figure/axes ourselves
    }

    if plot_kwargs_forest1 is None:
        plot_kwargs_forest1 = {}
    if plot_kwargs_forest2 is None:
        plot_kwargs_forest2 = {}

    kwargs_cloud_1 = {**base_plot_kwargs, **plot_kwargs_forest1}
    kwargs_cloud_2 = {**base_plot_kwargs, **plot_kwargs_forest2}

    if "title" not in plot_kwargs_forest1:
        plot_kwargs_forest1["title"] = "Acitve Loops"
    if "title" not in plot_kwargs_forest2:
        plot_kwargs_forest2["title"] = "Acitve Loops"

    # --- 6) Helpers to draw a single frame -----------------------------------
    def _draw_frame_at_time(t: float):
        # First forest
        ax_cloud_1.clear()
        forest1.plot_at_filtration(filt_val=t, ax=ax_cloud_1, **kwargs_cloud_1)
        ax_cloud_1.text(
            0.02, 0.98, rf"$\alpha = {t:.3g}$",
            transform=ax_cloud_1.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Second forest
        ax_cloud_2.clear()
        forest2.plot_at_filtration(filt_val=t, ax=ax_cloud_2, **kwargs_cloud_2)
        ax_cloud_2.text(
            0.02, 0.98, rf"$\alpha = {t:.3g}$",
            transform=ax_cloud_2.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    def _align_barcode_axes():
        # Make barcode axes have same left/right as their corresponding cloud axes
        for cloud_ax, bar_ax in ((ax_cloud_1, ax_bar_1), (ax_cloud_2, ax_bar_2)):
            cloud_pos = cloud_ax.get_position()
            bar_pos = bar_ax.get_position()
            # Keep bar's vertical position/height, but match horizontal start and width
            bar_ax.set_position([
                cloud_pos.x0,    # left
                bar_pos.y0,      # bottom
                cloud_pos.width, # width
                bar_pos.height,  # height
            ])

    # --- 7) Animation callbacks -----------------------------------------------
    def init():
        _draw_frame_at_time(frame_times[0])
        _align_barcode_axes()
        t_init = frame_times[0]
        barcode_line_1.set_xdata([t_init, t_init])
        barcode_line_2.set_xdata([t_init, t_init])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        _align_barcode_axes()
        barcode_line_1.set_xdata([t, t])
        barcode_line_2.set_xdata([t, t])
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        blit=False,
    )

    # --- 8) Optional save to disk ---------------------------------------------
    if filename is not None:
        fname = str(filename)
        ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
        if ext == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(fname, writer=writer, dpi=dpi)
        elif ext in {"gif", "gifv"}:
            try:
                from matplotlib.animation import PillowWriter
            except ImportError as e:
                raise RuntimeError(
                    "Saving as GIF requires Pillow. Install it with `pip install pillow`."
                ) from e
            writer = PillowWriter(fps=fps)
            anim.save(fname, writer=writer, dpi=dpi)
        else:
            # Fallback to default writer chosen by matplotlib
            anim.save(fname, dpi=dpi, fps=fps)

    return anim, fig