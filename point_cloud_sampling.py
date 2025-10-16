import numpy as np

def sample_noisy_star(n, spikes=5, amplitude=0.5, noise_std=0.01, seed=None, radius = 0.5):
    """
    Sample n random points from a star-shaped curve in 2D,
    perturbed by Gaussian noise.
    
    Parameters:
        n (int): Number of points to sample.
        spikes (int): Number of star spikes.
        amplitude (float): Spike strength (0 = circle).
        noise_std (float): Std. dev. of Gaussian noise.
        seed (int or None): Random seed.
    
    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    r0 = radius  # base radius (like circle with diameter 1)
    theta = np.random.uniform(0, 2*np.pi, n)

    # star shape: radial function
    r = r0 * (1 + amplitude * np.cos(spikes * theta))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)
    
    return np.column_stack((x_noisy, y_noisy))

def sample_noisy_circle(n, noise_std=0.01, seed=42, radius: float = 1):
    """
    Sample n random points on a circle with diameter 1,
    perturbed by Gaussian noise.
    
    Parameters:
        n (int): Number of points to sample.
        noise_std (float): Standard deviation of Gaussian noise.
        seed (int or None): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    r = radius  # radius of circle
    theta = np.random.uniform(0, 2 * np.pi, n)  # random angles
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)
    
    return np.column_stack((x_noisy, y_noisy))


def sample_noisy_ellipse(
    n: int,
    a: float = 1.5,
    b: float = 1.0,
    angle: float = 0.0,
    noise_std: float = 0.01,
    seed: int | None = 42,
    center: tuple[float, float] = (0.0, 0.0),
):
    """
    Sample n noisy points on the perimeter of an ellipse.

    The ellipse is defined by semi-axes a (x-axis) and b (y-axis), optionally
    rotated by 'angle' radians about its center, then perturbed by Gaussian noise.

    Parameters:
        n (int): Number of points to sample.
        a (float): Semi-major axis length along x before rotation.
        b (float): Semi-minor axis length along y before rotation.
        angle (float): Rotation of the ellipse in radians (counterclockwise).
        noise_std (float): Std. dev. of isotropic Gaussian noise added to (x, y).
        seed (int or None): Random seed for reproducibility.
        center (tuple): (cx, cy) center of the ellipse.

    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Parameterize ellipse: (a cos t, b sin t), t uniform in [0, 2π)
    # (Note: uniform in parameter, not exactly uniform in arclength.)
    t = np.random.uniform(0, 2 * np.pi, n)
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)

    # Rotate by 'angle'
    ca, sa = np.cos(angle), np.sin(angle)
    x = ca * x0 - sa * y0
    y = sa * x0 + ca * y0

    # Translate to center
    x += center[0]
    y += center[1]

    # Add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)

    return np.column_stack((x_noisy, y_noisy))


def sample_noisy_circle_with_tendril(
    n: int,
    noise_std: float = 0.01,
    seed: int | None = 42,
    radius: float = 1.0,
    tendril_length: float = 0.7,
    tendril_fraction: float = 0.15,
    tendril_angle: float = 0.0,
    tendril_width_deg: float = 8.0,
    radial_jitter: float = 0.02,
):
    """
    Sample n noisy points on a circle plus a thin inward tendril.

    The base shape is a circle of given radius. A fraction of points are drawn
    along a narrow angular sector ('tendril') that extends radially inward from
    the circle toward the center.

    Parameters:
        n (int): Number of points to sample.
        noise_std (float): Std. dev. of isotropic Gaussian noise added to (x, y).
        seed (int or None): Random seed for reproducibility.
        radius (float): Circle radius.
        tendril_length (float): How far inward the tendril reaches (in radius units).
                                Effective radius along tendril goes from r to r - tendril_length.
        tendril_fraction (float): Fraction of points to allocate to the tendril (0–1).
        tendril_angle (float): Central angle (radians) where the tendril is located.
        tendril_width_deg (float): Angular spread (degrees) around tendril_angle.
        radial_jitter (float): Extra radial noise (std dev) applied to tendril points only.

    Returns:
        np.ndarray: Array of shape (n, 2) with noisy (x, y) points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Split points between circle and tendril
    n_tendril = int(round(tendril_fraction * n))
    n_circle = n - n_tendril

    # --- Circle perimeter points ---
    theta_circle = np.random.uniform(0, 2 * np.pi, n_circle)
    x_c = radius * np.cos(theta_circle)
    y_c = radius * np.sin(theta_circle)

    # --- Tendril points ---
    # Narrow angular sector around tendril_angle
    width_rad = np.deg2rad(tendril_width_deg)
    theta_t = np.random.normal(loc=tendril_angle, scale=width_rad / 2.355, size=n_tendril)
    # (Use FWHM ≈ 2.355*σ so 'tendril_width_deg' roughly matches visible width.)

    # Radii move inward from r to r - tendril_length (thin path)
    t = np.random.uniform(0.0, 1.0, n_tendril)
    r_t = radius - tendril_length * t + np.random.normal(0, radial_jitter, n_tendril)
    r_t = np.clip(r_t, 0.0, radius)

    x_t = r_t * np.cos(theta_t)
    y_t = r_t * np.sin(theta_t)

    # Combine
    x = np.concatenate([x_c, x_t])
    y = np.concatenate([y_c, y_t])

    # Add overall Gaussian positional noise
    x_noisy = x + np.random.normal(0, noise_std, n)
    y_noisy = y + np.random.normal(0, noise_std, n)

    # Shuffle so tendril/circle points are mixed
    pts = np.column_stack((x_noisy, y_noisy))
    idx = np.random.permutation(n)
    return pts[idx]
