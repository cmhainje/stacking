"""Lupton RGB colorization and RA/Dec plotting for multi-wavelength HEALPix flux maps.

Expects tabular flux measurements of the form (flux, wavelength, healpix_index)
at a fixed nside, with six distinct wavelengths. The bottom three (shortest)
wavelengths are composited into one Lupton RGB image, the top three (longest)
into another.
"""

import healpy as hp
import numpy as np
from astropy.visualization import LuptonAsinhStretch, ManualInterval, make_lupton_rgb


def bin_healpix_flux(flux, wavelength, hp_idx, nside, wavelengths):
    """Build one full-sky HEALPix map per wavelength from tabular measurements.

    Multiple measurements landing in the same (wavelength, pixel) are averaged.
    Pixels with no measurements are set to NaN.

    Returns
    -------
    dict mapping each entry of `wavelengths` to a length-(12 * nside**2) array.
    """
    flux = np.asarray(flux, dtype=float)
    wavelength = np.asarray(wavelength)
    hp_idx = np.asarray(hp_idx, dtype=np.int64)
    npix = hp.nside2npix(nside)

    maps = {}
    for w in wavelengths:
        mask = wavelength == w
        sums = np.zeros(npix)
        counts = np.zeros(npix)
        np.add.at(sums, hp_idx[mask], flux[mask])
        np.add.at(counts, hp_idx[mask], 1)
        m = np.full(npix, np.nan)
        seen = counts > 0
        m[seen] = sums[seen] / counts[seen]
        maps[w] = m
    return maps


def lupton_rgb(image_r, image_g, image_b, vmin=0.0, vmax=None, stretch=5.0, Q=8.0):
    """Composite three maps into a Lupton-algorithm RGB image.

    Parameters
    ----------
    image_r, image_g, image_b : array-like
        Equal-shape flux maps (e.g. HEALPix arrays); NaNs are treated as 0.
    vmin, vmax : float or (3,) sequence, optional
        Black point / white point per channel. Scalars are broadcast to all
        three channels. `vmax=None` disables clipping on the bright end
        (only the asinh stretch controls saturation).
    stretch : float
        Linear stretch factor. Larger values compress the bright end less,
        pushing more of the dynamic range into the faint end.
    Q : float
        Asinh softening parameter. Small Q (~1-2) behaves close to linear
        near black; large Q (~10+) makes the transition to the asinh regime
        gentler, preserving color in bright saturated regions.
    """
    def _triplet(v):
        v = np.atleast_1d(v)
        return np.broadcast_to(v, (3,)) if v.size == 1 else v

    vmins, vmaxs = _triplet(vmin), _triplet(vmax if vmax is not None else [None] * 3)
    interval = [ManualInterval(vmin=vmins[i], vmax=vmaxs[i]) for i in range(3)]
    stretch_object = LuptonAsinhStretch(stretch=stretch, Q=Q)

    imgs = [np.nan_to_num(np.asarray(im, dtype=float)) for im in (image_r, image_g, image_b)]
    rgb = make_lupton_rgb(*imgs, interval=interval, stretch_object=stretch_object)
    return rgb.squeeze(axis=0) if rgb.ndim == 3 and rgb.shape[0] == 1 else rgb


def make_two_band_composites(maps, wavelengths, low_knobs=None, high_knobs=None):
    """Build the "bottom three" and "top three" wavelength RGB composites.

    Parameters
    ----------
    maps : dict {wavelength: healpix array}, as returned by `bin_healpix_flux`.
    wavelengths : sequence of the 6 wavelengths, in any order (will be sorted).
    low_knobs, high_knobs : dict, optional
        Keyword overrides passed to `lupton_rgb` (vmin, vmax, stretch, Q) for
        the short- and long-wavelength composites respectively.

    Returns
    -------
    (rgb_low, rgb_high) : two (npix, 3) uint8 arrays.
        rgb_low maps [w0, w1, w2] (shortest) -> [R, G, B]
        rgb_high maps [w3, w4, w5] (longest) -> [R, G, B]
    """
    ws = sorted(wavelengths)
    if len(ws) != 6:
        raise ValueError(f"expected 6 wavelengths, got {len(ws)}")

    low_knobs = low_knobs or {}
    high_knobs = high_knobs or {}
    rgb_low = lupton_rgb(maps[ws[2]], maps[ws[1]], maps[ws[0]], **low_knobs)
    rgb_high = lupton_rgb(maps[ws[5]], maps[ws[4]], maps[ws[3]], **high_knobs)
    return rgb_low, rgb_high


def plot_healpix_rgb_radec(
    rgb, nside, ax=None, only_seen=True, s=4.0, ra_center=180.0, title=None
):
    """Scatter-plot a per-pixel RGB HEALPix image against RA/Dec.

    Parameters
    ----------
    rgb : (npix, 3) array, uint8 or float in [0, 1].
    nside : the map's HEALPix nside (RING ordering assumed).
    ax : matplotlib Axes, optional. A new figure/axes is created if omitted.
    only_seen : if True, skip pixels that are exactly black (0, 0, 0) -
        i.e. pixels with no data in any channel - for a faster, cleaner plot.
    s : marker size passed to scatter.
    ra_center : RA (deg) to center the plot on; wraps RA into
        [ra_center - 180, ra_center + 180) so fields near RA=0 aren't split.
    title : optional axes title.
    """
    import matplotlib.pyplot as plt

    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    ra = ((ra - ra_center + 180.0) % 360.0) + (ra_center - 180.0)

    rgb = np.asarray(rgb)
    colors = rgb.astype(float) / 255.0 if rgb.dtype == np.uint8 else rgb

    if only_seen:
        keep = colors.sum(axis=1) > 0
        ra, dec, colors = ra[keep], dec[keep], colors[keep]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ra, dec, c=colors, s=s, marker="s", linewidths=0)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_xlim(ra_center + 180.0, ra_center - 180.0)  # RA increases to the left
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_two_band_composites(
    rgb_low, rgb_high, nside, low_title="short wavelengths", high_title="long wavelengths", **kwargs
):
    """Convenience: side-by-side RA/Dec plots for the two composites."""
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    plot_healpix_rgb_radec(rgb_low, nside, ax=ax0, title=low_title, **kwargs)
    plot_healpix_rgb_radec(rgb_high, nside, ax=ax1, title=high_title, **kwargs)
    fig.tight_layout()
    return fig, (ax0, ax1)
