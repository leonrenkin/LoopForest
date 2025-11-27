import math
from typing import Iterable, Tuple
from numpy.typing import NDArray
import numpy as np

_EPS = 1e-12

def _to_xy(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    """Convert input to an (N,2) float64 array and drop duplicated last=first."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like of x,y pairs")
    if len(P) < 3:
        raise ValueError("Need at least 3 distinct points for a polygon")
    if np.linalg.norm(P[0] - P[-1]) < _EPS:  # drop repeated closure point
        P = P[:-1].copy()
    return P

def polygon_length(points: Iterable[Tuple[float, float]]) -> float:
    """
    Perimeter (sum of edge lengths) of the closed polygonal loop.
    """
    P = _to_xy(points)
    diffs = np.roll(P, -1, axis=0) - P
    return float(np.linalg.norm(diffs, axis=1).sum())

def polygon_area(points: Iterable[Tuple[float, float]], signed: bool = False) -> float:
    """
    Shoelace area of the polygon. Positive for CCW, negative for CW (if signed=True).
    """
    P = _to_xy(points)
    x, y = P[:, 0], P[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    a = 0.5 * float(np.dot(x, ys) - np.dot(y, xs))
    return a if signed else abs(a)

def polygon_area_length_ratio(points: Iterable[Tuple[float, float]]) -> float:
    return polygon_area(points=points)/(polygon_length(points=points)**2)

def polygon_area_length_squared_ratio(points: Iterable[Tuple[float, float]]) -> float:
    return polygon_area(points=points)/polygon_length(points=points)

def polygon_length_area_ratio(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)/polygon_area(points=points)

def polygon_length_squared_area_ratio_normalized(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)**2/polygon_area(points=points) - 4*np.pi

def polygon_length_squared_area_ratio(points: Iterable[Tuple[float, float]]) -> float:
    return (polygon_length(points=points)**2)/polygon_area(points=points)

def total_curvature(points: Iterable[Tuple[float, float]],
                    enforce_ccw: bool = True,
                    ) -> float:
    """
    Total curvature of a simple closed polygonal loop:
        K_total = sum_i |kappa_i|
    where kappa_i is the signed exterior turning angle at vertex i.

    Definition at vertex i:
        a = P[i]   - P[i-1]   (incoming edge)
        b = P[i+1] - P[i]     (outgoing edge)
        kappa_i = atan2( cross(a,b), dot(a,b) ) ∈ (-π, π]

    Properties
    ----------
    - For a simple CCW convex polygon (e.g., regular n-gon), sum(kappa_i) ≈ +2π and
      total_curvature ≈ 2π.
    - For non-convex/star-shaped polygons (some kappa_i < 0), total_curvature > 2π.
    - Invariant to translation and rotation; scale-invariant (angles only).

    Parameters
    ----------
    points : sequence of (x,y) defining a simple loop (last connects to first).
    enforce_ccw : bool
        If True, reverse order when signed area < 0 (purely for the sum check).
    check : bool
        If True, verify that sum of signed kappa_i ≈ ±2π (simple closed curve).

    Returns
    -------
    K : float
        Total curvature (radians).
    """
    P = _to_xy(points)

    """  # Orient CCW if requested (helps interpret the winding check).
    signed_a = polygon_area(P, signed=True)
    if enforce_ccw and signed_a < 0:
        P = P[::-1].copy()
        signed_a = -signed_a """

    prevP = np.roll(P, 1, axis=0)
    nextP = np.roll(P, -1, axis=0)
    a = P - prevP
    b = nextP - P

    a_len = np.linalg.norm(a, axis=1)
    b_len = np.linalg.norm(b, axis=1)
    bad = (a_len < _EPS) | (b_len < _EPS)
    if np.any(bad):
        raise ValueError("Zero-length edge detected near vertices: "
                         f"{np.where(bad)[0].tolist()}")

    cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]     # z-component
    dot   = a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1]
    kappa = np.arctan2(cross, dot)                # signed exterior angle ∈ (-π, π]
    abs_kappa = np.abs(kappa)

    K = float(abs_kappa.sum())
    return K

def curvature_excess(points: Iterable[Tuple[float, float]]) -> float:
    """
    Excess total curvature beyond 2π, scaled by 2π:
        CE = K_total / (2π) - 1
    CE ≈ 0 for convex shapes (including circle-like); CE > 0 for star-shaped/non-convex.
    """
    K = total_curvature(points, enforce_ccw=True)
    return K *(1.0/ (2*math.pi) )- 1.0

def signed_chain_edge_length(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    CycleValueFunc for signed chains: total edge length.

    Parameters
    ----------
    rep
        A SignedChain-like object with an attribute
            rep.signed_simplices : iterable of (simplex, sign)
        where `simplex` is an iterable of vertex indices (e.g. (i, j)),
        and `sign` is typically ±1 (orientation).
    point_cloud : np.ndarray, shape (n_points, dim)
        Ambient point cloud.

    Returns
    -------
    float
        Sum of Euclidean lengths of all 1-simplices in the chain.

    Notes
    -----
    - Only simplices of length 2 (edges) are used.
    - The sign is ignored in the magnitude (we use abs(sign)), so this is
      "unsigned total length". If you want oriented total length, replace
      `abs(sign)` by `sign`.
    """
    total = 0.0

    for simplex, sign in signed_chain.signed_simplices:
        # Make sure we have exactly two vertices: an edge
        verts_idx = list(simplex)
        if len(verts_idx) != 2:
            continue

        p0 = point_cloud[verts_idx[0]]
        p1 = point_cloud[verts_idx[1]]
        length = float(np.linalg.norm(p1 - p0))

        # Use abs(sign) so orientation doesn't cancel length
        total += length

    return total

def constant_one_functional(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    return 1

def signed_chain_connected_components(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    return len( signed_chain.polyhedral_paths(point_cloud) )

def signed_chain_excess_connected_components(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    return len( signed_chain.polyhedral_paths(point_cloud) ) - 1

def signed_chain_area(signed_chain, point_cloud:  NDArray[np.float64]) -> float:
    """
    Compute Area by taking area of outer circle minus area of inner circles
    """

    paths = list( signed_chain.polyhedral_paths(point_cloud) )
    x_max_list = np.array([point_cloud[path, 0].max() for path in paths])
    index_max = np.argmax(x_max_list)

    total_area = 0

    for index, path in enumerate(paths):
        if len(path) <= 2:
            print(paths)
            raise ValueError("Paths too short")
        if index == index_max:
            total_area += polygon_area(point_cloud[path])
        else:
            total_area -= polygon_area(point_cloud[path])
    return total_area

def signed_chain_excess_curvature(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """Sums excess curvature of polyhedral paths"""
    paths = list( signed_chain.polyhedral_paths(point_cloud) )
    total = 0
    for path in paths:
        if len(path) <= 2:
            print(paths)
            raise ValueError("Paths too short")
        total += curvature_excess(point_cloud[path])

    return total