"""
"""

import numpy as np

from pyproj import Proj
from openquake.sub.misc.edge import _get_mean_longitude


def get_number_ruptures(omsh, rup_s, rup_d, f_strike=1, f_dip=1, wei=None):
    """
    Given a :class:`~openquake.hazardlib.geo.mesh.Mesh` instance and the size
    of a rupture (in terms of the number of rows and cols) it provides the
    number of ruptures admitted and the sum of their weights.

    :param omsh:
        A :class:`~openquake.hazardlib.geo.mesh.Mesh` instance describing the
        fault surface
    :param rup_s:
        Number of cols composing the rupture
    :param rup_d:
        Number of rows composing the rupture
    :param wei:
        Weights for each cell composing the fault surface
    :param f_strike:
        Floating distance along strike (multiple of sampling distance)
    :param f_dip:
        Floating distance along dip (multiple of sampling distance)
    """
    num_rup = 0
    wei_rup = []
    for i in np.arange(0, omsh.lons.shape[1] - rup_s, f_strike):
        for j in np.arange(0, omsh.lons.shape[0] - rup_d, f_dip):
            if (np.all(np.isfinite(omsh.lons[j:j + rup_d, i:i + rup_s]))):
                if wei is not None:
                    wei_rup.append(np.sum(wei[j:j + rup_d - 1,
                                              i:i + rup_s - 1]))
                num_rup += 1
    return num_rup


def get_ruptures(omsh, rup_s, rup_d, f_strike=1, f_dip=1, wei=None):
    """
    Given a :class:`~openquake.hazardlib.geo.mesh.Mesh` instance and the size
    of a rupture (in terms of the number of rows and cols) it yields all the
    possible ruptures admitted by the fault geometry.

    :param omsh:
        A :class:`~openquake.hazardlib.geo.mesh.Mesh` instance describing the
        fault surface
    :param rup_s:
        Number of cols composing the rupture
    :param rup_d:
        Number of rows composing the rupture
    :param wei:
        Weights for each cell composing the fault surface
    :param f_strike:
        Floating distance along strike (multiple of sampling distance)
    :param f_dip:
        Floating distance along dip (multiple of sampling distance)
    :returns:

    """
    if f_strike < 0:
        f_strike = int(np.floor(rup_s * abs(f_strike) + 1e-5))
        if f_strike < 1:
            f_strike = 1
    if f_dip < 0:
        f_dip = int(np.floor(rup_d * abs(f_dip) + 1e-5))
        if f_dip < 1:
            f_dip = 1
    for i in np.arange(0, omsh.lons.shape[1] - rup_s + 2, f_strike):
        for j in np.arange(0, omsh.lons.shape[0] - rup_d + 2, f_dip):
            if (np.all(np.isfinite(omsh.lons[j:j + rup_d, i:i + rup_s]))):
                yield ((omsh.lons[j:j + rup_d, i:i + rup_s],
                        omsh.lats[j:j + rup_d, i:i + rup_s],
                        omsh.depths[j:j + rup_d, i:i + rup_s]), j, i)


def get_weights(centroids, r, values, proj):
    """
    :param centroids:
        A :class:`~numpy.ndarray` instance with cardinality j x k x 3 where
        j and k corresponds to the number of cells along strike and along dip
        forming the rupture
    :param r:
        A :class:`~rtree.index.Index` instance for the location of the values
    :param values:
        A :class:`~numpy.ndarray` instance with lenght equal to the number of
        rows in the `centroids` matrix
    :param proj:
    :returns:

    """
    #
    # set the projection
    p = proj
    # projected centroids - projection shouldn't be an issue here as long as
    # we can get the nearest neighbour correctly
    cx, cy = p(centroids[:, :, 0].flatten(), centroids[:, :, 1].flatten())
    cx *= 1e-3
    cy *= 1e-3
    cz = centroids[:, :, 2].flatten()
    #
    # assign a weight to each centroid
    weights = np.zeros_like(cx)
    weights[:] = np.nan
    for i in range(0, len(cx)):
        if np.isfinite(cz[i]):
            idx = list(r.nearest((cx[i], cy[i], cz[i], cx[i], cy[i], cz[i]), 1,
                                 objects=False))
            weights[i] = values[idx[0]]
    #
    # reshape the weights
    weights = np.reshape(weights, (centroids.shape[0], centroids.shape[1]))
    return weights


def heron_formula(coords):
    """
    """
    pass


def get_mesh_area(mesh):
    """
    :param mesh:
        A :class:`numpy.ndarray` instance.
    """
    for j in range(0, mesh.shape[0]-1):
        for k in range(0, mesh.shape[1]-1):
            if (np.all(np.isfinite(mesh.depths[j:j+1, k:k+1]))):
                pass
                # calculate the area


def get_discrete_dimensions(area, sampling, aspr):
    """
    :param area:
    :param sampling:
    :param aspr:
    """
    #
    lng1 = np.ceil((area * aspr)**0.5/sampling)*sampling
    wdtA = np.ceil(lng1/aspr/sampling)*sampling
    wdtB = np.floor(lng1/aspr/sampling)*sampling

    lng2 = np.floor((area * aspr)**0.5/sampling)*sampling
    wdtC = np.ceil(lng2/aspr/sampling)*sampling
    wdtD = np.floor(lng2/aspr/sampling)*sampling
    #
    dff = 1e10
    lng = None
    wdt = None
    if abs(lng1*wdtA-area) < dff:
        lng = lng1
        wdt = wdtA
        dff = abs(lng1*wdtA-area)
    if abs(lng1*wdtB-area) < dff:
        lng = lng1
        wdt = wdtB
        dff = abs(lng1*wdtB-area)
    if abs(lng2*wdtC-area) < dff:
        lng = lng2
        wdt = wdtC
        dff = abs(lng2*wdtC-area)
    if abs(lng2*wdtD-area) < dff:
        lng = lng2
        wdt = wdtD
        dff = abs(lng2*wdtD-area)
    return lng, wdt
