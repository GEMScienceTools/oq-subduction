"""
"""

import re
import copy
import logging
import numpy as np

from pyproj import Proj, Geod

from openquake.sub.misc.profile import (_resample_profile,
                                        profiles_depth_alignment)

from openquake.hazardlib.geo.utils import plane_fit
from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.geodetic import azimuth
from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo.geodetic import npoints_towards

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EDGE_TOL = 0.2
PROF_TOL = 0.4
TOL = 0.5


def line_between_two_points(pnt1, pnt2):
    """
    :param numpy.ndarray pnt1:
    :param numpy.ndarray pnt2:
    """
    dcos = np.empty((3))
    dcos[0] = (pnt2[0] - pnt1[0])
    dcos[1] = (pnt2[1] - pnt1[1])
    dcos[2] = (pnt2[2] - pnt1[2])
    dcos = dcos * (sum(dcos**2))**-0.5
    return dcos


def _from_profiles_to_mesh(plist):
    """
    :param list plist:
        A list of lists. Each list contains three vectors of coordinates
    :returns:
        An array
    """
    mlen = 0.
    #
    # find the length of the longest profile
    for p in plist:
        mlen = max(mlen, len(p))
    #
    # initialise the mesh
    msh = np.full((mlen, len(plist), 3), np.nan)
    #
    # populate the mesh
    for i in range(0, mlen):
        for j, p in enumerate(plist):
            if len(p) > i:
                msh[i, j, 0] = p[0][i]
                msh[i, j, 1] = p[1][i]
                msh[i, j, 2] = p[2][i]
    return msh


def _rotate_vector(vect, rotax, angle):
    """
    Rotates a vector of a given angle using a given rotation axis using the
    Rodrigues formula.
    (see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

    :param numpy.ndarray vect:
    :param numpy.ndarray rotax:
    :param float angle:
    """
    vect = vect * (sum(vect**2))**-0.5
    rotax = rotax * (sum(rotax**2))**-0.5
    arad = np.radians(angle)
    t1 = vect * np.cos(arad)
    t2 = np.cross(rotax, vect) * np.sin(arad)
    t3 = rotax * np.dot(rotax, vect) * (1. - np.cos(arad))
    return t1 + t2 + t3


def _get_mean_longitude(tmp):
    """
    :param tmp:
        The vector containing the longitude values
    :returns:
        A float representing the mean longitude
    """
    if np.amax(tmp) - np.amin(tmp) > 100:
        tmp[tmp < 0] = 360. + tmp[tmp < 0]
        melo = np.mean(tmp)
        if melo > 180:
            melo = 360 - melo
        del tmp
    else:
        melo = np.mean(tmp)
    return melo


def create_faults(mesh, iedge, thickness, rot_angle, sampling):
    """
    Creates a list of profiles at a given angle from a mesh limiting the fault
    at the top. The fault is confined with a seismogenic layer with a thickness
    provided by the user.

    :param numpy.ndarray mesh:
        The mesh defining the top of the slab
    :param int iedge:
        ID of the edge to be used for the construction of the plane
    :param float thickness:
        The thickness [km] of the layer containing the fault
    :param float rot_angle:
        Rotation angle of the new fault (reference is the dip direction of the
        plane interpolation the slab surface)
    :param float sampling:
        The sampling distance used to create the profiles [km]
    :returns:
        A mesh representing the surface of the new fault source.
    """
    #
    # save mesh original shape
    shape = mesh[:, :, 0].shape
    #
    # get indexes of the edge
    idxs = np.nonzero(np.isfinite(mesh[iedge, :, 2]))
    # idxs = np.nonzero(np.isfinite(mesh[:, iedge, 2]))
    #
    # create a 3xn array with the points composing the mesh
    lld = np.array([mesh[:, :, 0].flatten('C'), mesh[:, :, 1].flatten('C'),
                    mesh[:, :, 2].flatten('C')]).T
    idx = np.isnan(lld[:, 0])
    assert np.nanmax(mesh[:, :, 2]) < 750.
    #
    # project the points using Lambert Conic Conformal - for the reference
    # meridian 'lon_0' we use the mean longitude of the mesh
    melo = _get_mean_longitude(lld[:, 0])
    p = Proj('+proj=lcc +lon_0={:f}'.format(melo))
    x, y = p(lld[:, 0], lld[:, 1])
    x = x / 1e3  # m -> km
    y = y / 1e3  # m -> km
    x[idx] = np.nan
    y[idx] = np.nan
    #
    # create a np.array with the same shape of the input 'mesh' but with
    # projected coordinates
    tmpx = np.reshape(x, shape,  order='C')
    tmpy = np.reshape(y, shape,  order='C')
    meshp = np.stack((tmpx, tmpy, mesh[:, :, 2]), axis=2)
    assert np.nanmax(meshp[:, :, 2]) < 750.
    #
    # check if the selected edge is continuous, otherwise split
    if np.all(np.diff(idxs) == 1):
        ilists = [list(idxs[0])]
    else:
        ilists = []
        cnt = 0
        tmp = []
        for i, idx in enumerate(list(idxs[0])):
            if i == 0:
                tmp.append(idx)
            else:
                if idx-tmp[-1] == 1:
                    tmp.append(idx)
                else:
                    ilists.append(tmp)
                    tmp = [idx]
                    cnt += 1
        if len(tmp) > 1:
            ilists.append(tmp)
    #
    # Remove single element lists
    for i, t in enumerate(ilists):
        if len(t) < 2:
            del ilists[i]
    #
    # plane fitting
    tmp = np.vstack((meshp[:, :, 0].flatten(),
                     meshp[:, :, 1].flatten(),
                     meshp[:, :, 2].flatten())).T
    idx = np.isfinite(tmp[:, 0])
    _, pppar = plane_fit(tmp[idx, :])
    #
    # process the edge
    dlt = 1
    rlow = iedge - dlt
    rupp = iedge + dlt + 1
    plist = []
    #
    # loop over 'segments' composing the edge
    for ilist in ilists:
        temp_plist = []
        #
        # loop over the indexes of the nodes composing the edge 'segment' and
        # for each point we create a new profile using the dip angle
        for i, ic in enumerate(ilist):
            #
            # initialise the indexes
            clow = ic - dlt
            cupp = ic + dlt + 1
            #
            # fixing indexes at the borders of the mesh
            if rlow < 0:
                rlow = 0
                rupp = rlow + dlt*2 + 1
            if clow < 0:
                clow = 0
                cupp = clow + dlt*2 + 1
            if rupp >= meshp.shape[0]:
                rupp = meshp.shape[0] - 1
                rlow = max(0, rupp - (dlt*2 + 1))
            if cupp >= meshp.shape[1]:
                cupp = meshp.shape[1] - 1
                clow = cupp - (dlt*2 + 1)
            #
            # coordinates subset
            tmp = np.vstack((meshp[rlow:rupp, clow:cupp, 0].flatten(),
                             meshp[rlow:rupp, clow:cupp, 1].flatten(),
                             meshp[rlow:rupp, clow:cupp, 2].flatten())).T
            #
            # interpolate the plane
            ii = np.isfinite(tmp[:, 2])
            if np.sum(ii) > 4:
                try:
                    _, ppar = plane_fit(tmp[ii, :])
                except:
                    raise ValueError('Plane interpolation failed')
            else:
                ppar = pppar
            #
            # vertical plane with the same strike
            vertical_plane = np.array([ppar[0], ppar[1], 0])
            vertical_plane = vertical_plane / (sum(vertical_plane**2))**.5
            #
            # strike direction
            stk = np.cross(ppar, vertical_plane)
            stk = stk / (sum(stk**2.))**0.5
            #
            # compute the vector on the plane defining the steepest direction
            # https://goo.gl/UtKJxe
            dip = np.cross(ppar, np.cross([0, 0, -1], ppar))
            dip = dip / (sum(dip**2.))**0.5
            #
            # rotate the dip of the angle provided by the user. Note that the
            # rotation follows the right hand rule. The rotation axis is the
            # strike
            dirc = _rotate_vector(dip, stk, rot_angle)
            #
            # compute the points composing the new surface. The new surface
            # is at a distance 'slab_tickness' below the original surface in a
            # direction perpendicular to the fitted planes
            corr = -1
            dstances = np.arange(0, thickness+0.05*sampling, sampling)
            xls = meshp[iedge, ic, 0] + corr * dstances * dirc[0]
            yls = meshp[iedge, ic, 1] + corr * dstances * dirc[1]
            zls = meshp[iedge, ic, 2] + corr * dstances * dirc[2]
            #
            # back-conversion to geographic coordinates
            llo, lla = p(xls*1e3, yls*1e3, inverse=True)
            #
            # Update the profile list
            line = Line([Point(x, y, z) for x, y, z in zip(llo, lla, zls)])
            temp_plist.append(line)
        #
        # updating the list of profiles
        if len(temp_plist) > 1:
            plist.append(temp_plist)
    #
    # return the list of profiles groups. Each group is a set of lines
    return plist


def get_coords(line):
    tmp = []
    for p in line.points:
        if p is not None:
            p.longitude = p.longitude+360 if p.longitude<0 else p.longitude
            tmp.append([p.longitude, p.latitude, p.depth])
    return tmp


def create_from_profiles(profiles, profile_sd, edge_sd, align=False):
    """
    This creates a mesh from a set of profiles

    :param list profiles:
        A list of :class:`openquake.hazardlib.geo.Line.line` instances
    :param float profile_sd:
        The sampling distance along the profiles
    :param edge_sd:
        The sampling distance along the edges
    :param align:
    :returns:
        A :class:`numpy.ndarray` instance with the coordinates of the mesh
    """
    #
    # resample profiles
    rprofiles = []
    for prf in profiles:
        rprofiles.append(_resample_profile(prf, profile_sd))
    tmps = 'Completed reading ({:d} loaded)'.format(len(rprofiles))
    logging.info(tmps)
    #
    # set the reference profile i.e. the longest one
    ref_idx = None
    max_length = -1e10
    for idx, prf in enumerate(rprofiles):
        length = prf.get_length()
        if length > max_length:
            max_length = length
            ref_idx = idx
    if ref_idx is not None:
        logging.info('Reference profile is # {:d}'.format(ref_idx))
    else:
        tmps = 'Reference profile undefined. # profiles: {:d}'
        logging.info(tmps.format(len(rprofiles)))
    #
    # -- CHECK --
    # check that in each profile the points are equally spaced
    for pro in rprofiles:
        pnts = [(pnt.longitude, pnt.latitude, pnt.depth) for pnt in pro.points]
        pnts = np.array(pnts)
        dst = distance(pnts[:-1, 0], pnts[:-1, 1], pnts[:-1, 2],
                       pnts[1:, 0], pnts[1:, 1], pnts[1:, 2])
        np.testing.assert_allclose(dst, profile_sd, rtol=1.)
    #
    # find the delta needed to align profiles if requested
    shift = np.zeros(len(rprofiles)-1)
    if align is True:
        for i in range(0, len(rprofiles)-1):
            shift[i] = profiles_depth_alignment(rprofiles[i], rprofiles[i+1])
    shift = np.array([0] + list(shift))
    #
    # find the maximum back-shift
    ccsum = [shift[0]]
    for i in range(1, len(shift)):
        ccsum.append(shift[i] + ccsum[i-1])
    add = ccsum - min(ccsum)
    #
    # create resampled profiles. Now the profiles should be all aligned from
    # the top
    rprof = []
    maxnum = 0
    for i, pro in enumerate(rprofiles):
        j = int(add[i])
        coo = get_coords(pro)
        tmp = [[np.nan, np.nan, np.nan] for a in range(0, j)]
        if len(tmp):
            points = tmp + coo
        else:
            points = coo
        rprof.append(points)
        maxnum = max(maxnum, len(rprof[-1]))
    logging.info('Completed creation of resampled profiles')
    #
    # Now profiles will have the same number of samples (some of them can be
    # nan)
    for i, pro in enumerate(rprof):
        while len(pro) < maxnum:
            pro.append([np.nan, np.nan, np.nan])
        rprof[i] = np.array(pro)
    #
    # create edges
    prfr = get_mesh(rprof, ref_idx, edge_sd)
    logging.info('Completed creation of resampled profiles')
    #
    # create the mesh
    if ref_idx > 0:
        prfl = get_mesh_back(rprof, ref_idx, edge_sd)
    else:
        prfl = []
    prf = prfl + prfr
    msh = np.array(prf)
    #
    # checks
    """
    for i in range(0, msh.shape[0]-1):
        for j in range(0, msh.shape[1]-1):
            if np.all(np.isfinite(msh[i:i+1, j, 2])):
                d = distance(msh[i, j, 0], msh[i, j, 1], msh[i, j, 2],
                             msh[i+1, j, 0], msh[i+1, j, 1], msh[i+1, j, 2])
                if abs(d-profile_sd) > TOL*profile_sd:
                    print(d, abs(d-profile_sd), TOL*profile_sd)
                    raise ValueError('')
    """
    #
    # --------------------------------------------------------------------------
    if 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for pro in profiles:
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], 'x--b', markersize=2)
        for i, tmp in enumerate(rprof):
            idx = np.isfinite(tmp[:, 0])
            iii = np.nonzero(idx)[0][0]
            ax.plot(tmp[idx, 0], tmp[idx, 1], tmp[idx, 2], '^-r', markersize=5)
            ax.text(tmp[iii, 0], tmp[iii, 1], tmp[iii, 2], '{:d}'.format(i))

        # for all edges
        for j in range(len(prf[0])-1):
            # for all profiles
            for k in range(len(prf)-1):
                # plotting profiles
                if (np.all(np.isfinite(prf[k][j])) and
                        np.all(np.isfinite(prf[k+1][j]))):
                    pa = prf[k][j]
                    pb = prf[k+1][j]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                            [pa[2], pb[2]], '-', color='cyan')
                # plotting edges
                if (np.all(np.isfinite(prf[k][j])) and
                        np.all(np.isfinite(prf[k][j+1]))):
                    pa = prf[k][j]
                    pb = prf[k][j+1]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                            [pa[2], pb[2]], '-g')
        ax.invert_zaxis()
        ax.view_init(50, 55)
        plt.show()
    # --------------------------------------------------------------------------
    #
    # convert from profiles to edges
    msh = msh.swapaxes(0, 1)
    return msh


def get_mesh_back(pfs, rfi, sd):
    """
    Compute resampled profiles in the backward direction from the reference
    profile and creates the portion of the mesh 'before' the reference profile.

    :param list pfs:
        Original profiles. Each profile is a :class:`numpy.ndarray` instance
        with 3 columns and as many rows as the number of points included\
    :param int rfi:
        Index of the reference profile
    :param sd:
        Sampling distance [in km]
    :returns:

    """
    tmps = 'Number of profiles: {:d}'
    logging.info(tmps.format(len(pfs)))
    #
    # projection
    g = Geod(ellps='WGS84')
    #
    # initialize residual distance and last index lists
    rdist = [0 for _ in range(0, len(pfs[0]))]
    laidx = [0 for _ in range(0, len(pfs[0]))]
    #
    # create list containing the new profiles
    npr = list([copy.deepcopy(pfs[rfi])])
    #
    # run for all the profiles from the reference one backward
    for i in range(rfi, 0, -1):
        #
        # profiles
        pr = pfs[i-1]
        pl = pfs[i]
        #
        # point in common on the two profiles
        cmm = np.logical_and(np.isfinite(pr[:, 2]), np.isfinite(pl[:, 2]))
        #
        # update last index
        cmmi = np.nonzero(cmm)[0].astype(int)
        mxx = 0
        for ll in laidx:
            if ll is not None:
                mxx = max(mxx, ll)
        #
        # update the last index
        for x in range(0, len(pr[:, 2])):
            if x in cmmi and laidx[x] is None:
                iii = []
                for li, lv in enumerate(laidx):
                    if lv is not None:
                        iii.append(li)
                iii = np.array(iii)
                minidx = np.argmin(abs(iii-x))
                laidx[x] = mxx
                rdist[x] = rdist[minidx]
            elif x not in cmmi:
                laidx[x] = None
                rdist[x] = 0
        #
        # loop over the points in common between the two profiles
        for k in list(np.nonzero(cmm)[0]):
            #
            # compute azimuth and horizontal distance
            az12, az21, hdist = g.inv(pl[k, 0], pl[k, 1], pr[k, 0], pr[k, 1])
            hdist /= 1e3
            vdist = pr[k, 2] - pl[k, 2]
            tdist = (vdist**2 + hdist**2)**.5
            ndists = int(np.floor((tdist+rdist[k])/sd))
            #
            # computing distance between adjacent points in two consecutive
            # profiles
            dd = distance(pl[k, 0], pl[k, 1], pl[k, 2],
                          pr[k, 0], pr[k, 1], pl[k, 2])
            #
            # Checking difference between computed and expected distances
            # if abs(dd-tdist) > TOL*tdist:
            #     print('Distances:', dd, tdist)
            #     raise ValueError('')
            #
            # adding new points along edge with index k
            for j, dst in enumerate(range(ndists)):
                #
                # add new profile
                if len(npr)-1 < laidx[k]+1:
                    npr = add_empy_profile(npr)
                #
                # fix distance
                tmp = (j+1)*sd - rdist[k]
                lo, la, _ = g.fwd(pl[k, 0], pl[k, 1], az12,
                                  tmp*hdist/tdist*1e3)
                lo = lo+360 if lo<0 else lo
                de = pl[k, 2] + tmp*vdist/hdist
                npr[laidx[k]+1][k] = [lo, la, de]

                if (k > 0 and np.all(np.isfinite(npr[laidx[k]+1][k])) and
                        np.all(np.isfinite(npr[laidx[k]][k]))):

                    p1 = npr[laidx[k]][k]
                    p2 = npr[laidx[k]+1][k]
                    d = distance(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])

                    # print(tmp, sd)
                    # print(d, ((tmp*hdist/tdist)**2+de**2)**.5)
                    #
                    # >>> TOLERANCE
                    if abs(d-sd) > TOL*sd:
                        tmpf = 'd: {:f} diff: {:f} tol: {:f} sd:{:f}'
                        tmpf += '\nresidual: {:f}'
                        tmps = tmpf.format(d, d-sd,  TOL*sd, sd, rdist[k])
                        logging.warning(tmps)
                        #
                        # plotting
                        if 1:
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            # profiles
                            for pro in pfs:
                                tmp = [[p[0], p[1], p[2]] for p in pro]
                                tmp = np.array(tmp)
                                tmp[:,0] = ([x+360 if x<0 else x
                                             for x in tmp[:,0]])
                                ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2],
                                        'x--b', markersize=2, label='original')
                            p1[0] = p1[0]+360 if p1[0]<0 else p1[0]
                            p2[0] = p2[0]+360 if p2[0]<0 else p2[0]
                            # new profiles
                            for pro in npr:
                                tmp = [[p[0], p[1], p[2]] for p in pro]
                                tmp = np.array(tmp)
                                ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2],
                                        'x--r', markersize=2)
                            ax.plot([p1[0]], [p1[1]], [p1[2]], 'og')
                            ax.plot([p2[0]], [p2[1]], [p2[2]], 'og')

                            ax.set_xlim([min(p1[0], p2[0])-.5,
                                         max(p1[0], p2[0])+.5])
                            ax.set_ylim([min(p1[1], p2[1])-.5,
                                         max(p1[1], p2[1])+.5])
                            ax.set_zlim([min(p1[2], p2[2])-5,
                                         max(p1[2], p2[2])+5])
                            ax.invert_zaxis()
                            plt.legend()
                            ax.view_init(50, 55)
                            plt.show()

                        raise ValueError('')

                laidx[k] += 1
            rdist[k] = tdist - sd*ndists + rdist[k]
            assert rdist[k] < sd

    if False:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # profiles
        for pro in pfs:
            tmp = [[p[0], p[1], p[2]] for p in pro]
            tmp = np.array(tmp)
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2],
                    'x--b', markersize=2)
        # new profiles
        for pro in npr:
            tmp = [[p[0], p[1], p[2]] for p in pro]
            tmp = np.array(tmp)
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2],
                    'x--r', markersize=2)
        ax.view_init(50, 55)
        plt.show()

    tmp = []
    for i in range(len(npr)-1, 0, -1):
        tmp.append(npr[i])

    return tmp


def get_mesh(pfs, rfi, sd):
    """
    """
    g = Geod(ellps='WGS84')
    #
    # residual distance, last index
    rdist = [0 for _ in range(0, len(pfs[0]))]
    laidx = [0 for _ in range(0, len(pfs[0]))]
    #
    # new profiles
    npr = list([copy.deepcopy(pfs[rfi])])
    #
    # run for all the profiles 'after' the reference one
    for i in range(rfi, len(pfs)-1):
        #
        # profiles
        pr = pfs[i+1]
        pl = pfs[i]

        for ii in range(0,len(pl)):
            ptmp = pl[ii][0]
            ptmp = ptmp+360 if ptmp<0 else ptmp
            pl[ii][0] = ptmp

        #
        # point in common on the two profiles
        cmm = np.logical_and(np.isfinite(pr[:, 2]), np.isfinite(pl[:, 2]))
        #
        # update last profile index
        cmmi = np.nonzero(cmm)[0].astype(int)
        mxx = 0
        for ll in laidx:
            if ll is not None:
                mxx = max(mxx, ll)
        #
        #
        for x in range(0, len(pr[:, 2])):
            # this edge is in common between the last and the current profiles
            #
            if x in cmmi and laidx[x] is None:
                iii = []
                for li, lv in enumerate(laidx):
                    if lv is not None:
                        iii.append(li)
                iii = np.array(iii)
                minidx = np.argmin(abs(iii-x))
                laidx[x] = mxx
                rdist[x] = rdist[minidx]
            elif x not in cmmi:
                laidx[x] = None
                rdist[x] = 0
        #
        # loop over profiles
        for k in list(np.nonzero(cmm)[0]):
            #
            #
            az12, az21, hdist = g.inv(pl[k, 0], pl[k, 1], pr[k, 0], pr[k, 1])
            hdist /= 1e3
            vdist = pr[k, 2] - pl[k, 2]
            tdist = (vdist**2 + hdist**2)**.5
            ndists = int(np.floor((tdist+rdist[k])/sd))
            #
            # checking distance calculation
            dd = distance(pl[k, 0], pl[k, 1], pl[k, 2],
                          pr[k, 0], pr[k, 1], pl[k, 2])
            # >>> TOLERANCE
            if abs(dd-tdist) > 0.5*tdist:
                tmps = 'Distances: {:f} {:f}'
                raise ValueError(tmps.format(dd, tdist))
            #
            # adding new points along the edge with index k
            for j, dst in enumerate(range(ndists)):
                #
                # add new profile
                if len(npr)-1 < laidx[k]+1:
                    npr = add_empy_profile(npr)
                #
                # fix distance
                tmp = (j+1)*sd - rdist[k]
                lo, la, _ = g.fwd(pl[k, 0], pl[k, 1], az12,
                                  tmp*hdist/tdist*1e3)
                lo = lo+360 if lo < 0 else lo
                de = pl[k, 2] + tmp*vdist/hdist
                npr[laidx[k]+1][k] = [lo, la, de]

                if (k > 0 and np.all(np.isfinite(npr[laidx[k]+1][k])) and
                        np.all(np.isfinite(npr[laidx[k]][k]))):

                    p1 = npr[laidx[k]][k]
                    p2 = npr[laidx[k]+1][k]
                    d = distance(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])

                    # >>> TOLERANCE
                    if abs(d-sd) > TOL*sd:
                        tmpf = 'd: {:f} diff: {:f} tol: {:f} sd:{:f}'
                        tmpf += '\nresidual: {:f}'
                        tmps = tmpf.format(d, d-sd,  TOL*sd, sd, rdist[k])
                        logging.warning(tmps)
                        #
                        # plotting
                        if 1:
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            for ipro, pro in enumerate(pfs):
                                tmp = [[p[0], p[1], p[2]] for p in pro]
                                tmp = np.array(tmp)
                                tmplon = tmp[:,0]
                                tmplon = ([x+360 if x<0 else x for x in tmplon])
                                tmplon0 = tmplon[0]
                                ax.plot(tmplon, tmp[:, 1], tmp[:, 2],
                                        'x--b', markersize=2)
                                ax.text(tmplon0, tmp[0, 1], tmp[0, 2],
                                        '{:d}'.format(ipro))
                            for pro in npr:
                                tmp = [[p[0], p[1], p[2]] for p in pro]
                                tmp = np.array(tmp)
                                tmplon = tmp[:,0]
                                tmplon = ([x+360 if x<0 else x for x in tmplon])
                                ax.plot(tmplon, tmp[:, 1], tmp[:, 2],
                                        'x--r', markersize=2)
                            p1[0] = p1[0]+360 if p1[0]<0 else p1[0]
                            p2[0] = p2[0]+360 if p2[0]<0 else p2[0]
                            ax.plot([p1[0]], [p1[1]], [p1[2]], 'og')
                            ax.plot([p2[0]], [p2[1]], [p2[2]], 'og')
                            ax.invert_zaxis()
                            ax.view_init(50, 55)
                            plt.show()
                        raise ValueError('')
                laidx[k] += 1
            rdist[k] = tdist - sd*ndists + rdist[k]
            assert rdist[k] < sd
    return npr


def add_empy_profile(npr, idx=-1):
    tmp = [[np.nan, np.nan, np.nan] for _ in range(len(npr[0]))]
    if idx == -1:
        npr = npr + [tmp]
    elif idx == 0:
        npr = [tmp] + npr
    else:
        ValueError('Undefined option')
    #
    # check that profiles have the same lenght
    for i in range(0, len(npr)-1):
        assert len(npr[i]) == len(npr[i+1])

    return npr


def _read_edge(filename):
    """
    :param filename:
        The name of the file with prefix 'edge'
        specifing the geometry of the top of the slab
    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)


def _resample_edge_with_direction(edge, sampling_dist, reference_idx,
                                  direct=+1):
    """
    :param edge:
    :param sampling_dist:
    :param reference_idx:
    :param direct:
    """
    #
    # checking that the increment is either 1 or -1
    assert abs(direct) == 1
    #
    # create three lists: one with longitude, one with latitude and one with
    # depth
    lo = [pnt.longitude for pnt in edge.points]
    la = [pnt.latitude for pnt in edge.points]
    de = [pnt.depth for pnt in edge.points]
    #
    # initialise the variable used to store the cumulated distance
    cdist = 0.
    #
    # initialise the list with the resampled nodes
    idx = reference_idx
    resampled_cs = [Point(lo[idx], la[idx], de[idx])]
    #
    # set the starting point
    slo = lo[idx]
    sla = la[idx]
    sde = de[idx]
    #
    # get the azimuth of the first segment on the edge in the given direction
    azim = azimuth(lo[idx], la[idx], lo[idx+direct], la[idx+direct])
    #
    # resampling
    old_dst = 1.e10
    while 1:
        #
        # this is a sanity check
        assert idx <= len(lo)-1
        #
        # check loop exit condition
        if direct > 0 and idx > len(lo)-1:
            break
        if direct < 0 and idx < 1:
            break
        #
        # compute the distance between the starting point and the next point
        # on the profile
        segment_len = distance(slo, sla, sde, lo[idx+direct], la[idx+direct],
                               de[idx+direct])
        #
        # search for the point
        if cdist+segment_len > sampling_dist:
            #
            # check
            if segment_len > old_dst:
                print(segment_len, '>', old_dst)
                raise ValueError('The segment length is increasing')
            else:
                old_dst = segment_len
            #
            # this is the lenght of the last segment-fraction needed to
            # obtain the sampling distance
            delta = sampling_dist - cdist
            #
            # compute the slope of the last segment and its horizontal length.
            # we need to manage the case of a vertical segment TODO
            segment_hlen = distance(slo, sla, 0., lo[idx+direct],
                                    la[idx+direct], 0.)
            segment_slope = np.arctan((de[idx+direct] - sde) / segment_hlen)
            #
            # horizontal and vertical lenght of delta
            delta_v = delta * np.sin(segment_slope)
            delta_h = delta * np.cos(segment_slope)
            #
            # add a new point to the cross section
            pnts = npoints_towards(slo, sla, sde, azim, delta_h, delta_v, 2)
            #
            # update the starting point
            slo = pnts[0][-1]
            sla = pnts[1][-1]
            sde = pnts[2][-1]
            #
            # checking distance between the reference point and latest point
            # included in the resampled section
            pnt = resampled_cs[-1]
            checkd = distance(slo, sla, sde, pnt.longitude, pnt.latitude,
                              pnt.depth)
            # >>> TOLERANCE
            if (cdist < 1e-2 and
                    abs(checkd - sampling_dist) > 0.05*sampling_dist):
                print(checkd, sampling_dist)
                msg = 'Segment distance different than sampling dst'
                raise ValueError(msg)
            #
            # updating the resample cross-section
            resampled_cs.append(Point(slo, sla, sde))
            #
            #
            tot = distance(lo[idx], la[idx], de[idx], lo[idx+direct],
                           la[idx+direct], de[idx+direct])
            downd = distance(slo, sla, sde, lo[idx], la[idx], de[idx])
            upd = distance(slo, sla, sde, lo[idx+direct], la[idx+direct],
                           de[idx+direct])

            # >>> TOLERANCE
            if abs(tot - (downd + upd)) > tot*0.05:
                print('     upd, downd, tot', upd, downd, tot)
                print(abs(tot - (downd + upd)))
                raise ValueError('Distances are not matching')
            #
            # reset the cumulative distance
            cdist = 0.
        else:
            # print('aa', cdist, segment_len, sampling_dist)
            # print('  ', idx, len(lo)-1, direct)
            #
            #
            old_dst = 1.e10

            cdist += segment_len
            idx += direct
            slo = lo[idx]
            sla = la[idx]
            sde = de[idx]
            #
            # get the azimuth of the profile
            if idx < len(lo)-1:
                azim = azimuth(lo[idx], la[idx],
                               lo[idx+direct], la[idx+direct])
            else:
                break
    #
    #
    return resampled_cs


def _resample_edge(edge, sampling_dist, reference_idx):
    """
    :param line:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :param sampling_dist:
        A scalar definining the distance used to sample the profile
    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    up = []
    lo = []
    #
    # if the reference index is lower then the maximum number of points
    # composing the edge we resample updward
    if reference_idx < len(edge)-1:
        up = _resample_edge_with_direction(edge, sampling_dist, reference_idx,
                                           direct=+1)
    # if the reference index is greater then 0 we resample downward
    if reference_idx > 0:
        lo = _resample_edge_with_direction(edge, sampling_dist, reference_idx,
                                           direct=-1)
        lo = lo[::-1]
    #
    # create the final list of points
    if reference_idx < len(edge)-1 and reference_idx > 0:
        pnts = lo[:-1] + up
    elif reference_idx == 0:
        pnts = up
    else:
        pnts = lo
    #
    # return results
    if len(pnts) > 1:
        return Line(pnts), len(lo), len(up)
    else:
        return None, None, None
