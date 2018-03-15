#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import h5py
import pickle
import numpy as np
import rtree
import logging
import configparser

from pyproj import Proj

from openquake.sub.plotting.tools import plot_mesh

from openquake.sub.misc.edge import create_from_profiles
from openquake.sub.quad.msh import create_lower_surface_mesh
from openquake.sub.grid3d import Grid3d
from openquake.sub.misc.profile import _read_profiles
from openquake.sub.misc.utils import (get_min_max, create_inslab_meshes,
                                      get_centroids)
from openquake.sub.slab.rupture_utils import (get_discrete_dimensions,
                                              get_ruptures, get_weights)

from openquake.baselib import sap
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.scalerel import get_available_scalerel
from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hazardlib.geo.surface.gridded import GriddedSurface

from oqmbt.tools.smooth3d import Smoothing3D

def get_catalogue(cat_pickle_fname, treg_filename, label):
    """
    :param cat_pickle_fname:
    :param treg_filename:
    :param label:
    """
    #
    # loading TR
    if treg_filename is not None:
        f = h5py.File(treg_filename, 'r')
        tr = f[label][:]
        f.close()
    #
    # loading the catalogue
    catalogue = pickle.load(open(cat_pickle_fname, 'rb'))
    catalogue.sort_catalogue_chronologically()
    #
    # if a label and a TR are provided we filter the catalogue
    if treg_filename is not None:
        selector = CatalogueSelector(catalogue, create_copy=False)
        catalogue = selector.select_catalogue(tr)
    return catalogue


def smoothing(mlo, mla, mde, catalogue, hspa, vspa, fname):
    """
    :param mlo:
    :param mla:
    :param mde:
    :param catalogue:
        An earthquake catalogue in the hmtk format
    :param hspa:
        Horizontal spacing [in km]
    :param vspa:
        Vertical spacing [in km]
    :param fname:
        Name of the hdf5 where to store the results
    :returns:
        A tuple including the values on the grid provided and the smoothing
        object
    """
    #
    # create mesh with the 3D grid of points and complete the smoothing
    mesh = Mesh(mlo, mla, mde)
    #
    # smoothing
    smooth = Smoothing3D(catalogue, mesh, hspa, vspa)
    values = smooth.gaussian(bffer=40, sigmas=[40, 20])
    #
    #
    if os.path.exists(fname):
        os.remove(fname)
    fh5 = h5py.File(fname, 'w')
    fh5.create_dataset('values', data=values)
    fh5.create_dataset('lons', data=mesh.lons)
    fh5.create_dataset('lats', data=mesh.lats)
    fh5.create_dataset('deps', data=mesh.depths)
    fh5.close()
    #
    return values, smooth


def spatial_index(smooth):
    """
    :param smooth:
        An instance of the :class:``
    """

    def _generator(mesh, p):
        """
        This is a generator function used to quickly populate the spatial
        index
        """
        for cnt, (lon, lat, dep) in enumerate(zip(mesh.lons.flatten(),
                                                  mesh.lats.flatten(),
                                                  mesh.depths.flatten())):
            x, y = p(lon, lat)
            x /= 1e3
            y /= 1e3
            yield (cnt, (x, y, dep, x, y, dep), 1)
    #
    # Setting rtree properties
    prop = rtree.index.Property()
    prop.dimension = 3
    #
    # Set the geographic projection
    lons = smooth.mesh.lons.flatten()
    p = Proj('+proj=lcc +lon_0={:f}'.format(np.mean(lons)))
    #
    # Create the spatial index for the grid mesh
    r = rtree.index.Index(_generator(smooth.mesh, p), properties=prop)
    # return the 3D spatial index - note that the index is in projected
    # coordinates
    return r, p


def create_ruptures(mfd, dips, sampling, msr, asprs, float_strike, float_dip,
                    r, values, oms, tspan, hdf5_filename, uniform_fraction,
                    proj):
    """
    Create inslab ruptures using an MFD, a time span. The dictionary 'oms'
    contains lists of profiles for various values of dip. The ruptures are
    floated on each virtual fault created from a set of profiles.

    :param mfd:
    :param dips:
    :param sampling:
    :param msr:
    :param asprs:
    :param float_strike:
    :param float_dip:
    :param r:
    :param values:
    :param oms:
        A dictionary. Values of dip are used as keys while values of the
        dictionary are list of lists. Every list contains one or several
        :class:`openquake.hazardlib.geo.line.Line` instances each one
        corresponding to a 3D profile.
    :param tspan:
        Time span [in yr]
    :param hdf5_filename:
        Name of the hdf5 file where to store the ruptures
    :param uniform_fraction:
        Fraction of the overall rate for a given magnitude bin to be
        distributed uniformly to all the ruptures for the same mag bin.
    """

    fh5 = h5py.File(hdf5_filename, 'a')
    grp_inslab = fh5.create_group('inslab')

    allrup = {}
    iscnt = 0
    for dip in dips:
        for mi, lines in enumerate(oms[dip]):
            #
            # filter out small surfaces
            if len(lines) < 3:
                continue
            #
            # create in-slab virtual fault - `lines` is the list of profiles
            # to be used for the construction of the virtual fault surface
            smsh = create_from_profiles(lines, profile_sd=sampling,
                                        edge_sd=sampling)
            omsh = Mesh(smsh[:, :, 0], smsh[:, :, 1], smsh[:, :, 2])
            #
            # store data in the hdf5 file
            grp_inslab.create_dataset('{:09d}'.format(iscnt), data=smsh)
            #
            # get centroids for a given virtual fault surface
            ccc = get_centroids(smsh[:, :, 0], smsh[:, :, 1], smsh[:, :, 2])
            #
            # get weights - this assigns to each cell centroid the weight of
            # the closest node in the values array
            weights = get_weights(ccc, r, values, proj)
            #
            # loop over magnitudes
            for mag, _ in mfd.get_annual_occurrence_rates():
                area = msr.get_median_area(mag=mag, rake=-90)
                rups = []
                for aspr in asprs:
                    #
                    # IMPORTANT: the sampling here must be consistent with
                    # what we use for the construction of the mesh
                    lng, wdt = get_discrete_dimensions(area, sampling, aspr)
                    #
                    # rupture lenght and rupture width as multiples of the
                    # mesh sampling distance
                    rup_len = int(lng/sampling) + 1
                    rup_wid = int(wdt/sampling) + 1

                    if rup_len < 2 or rup_wid < 2:
                        continue

                    #
                    # get_ruptures
                    for rup, rl, cl in get_ruptures(omsh, rup_len, rup_wid,
                                                    f_strike=float_strike,
                                                    f_dip=float_dip):
                        w = weights[cl:rup_len-1, rl:rup_wid-1]
                        i = np.isfinite(w)
                        #
                        # normalize the weight using the aspect ratio weight
                        wsum = sum(w[i])/asprs[aspr]
                        #
                        # create the gridded surface
                        if len(rup[0]):
                            srfc = GriddedSurface(Mesh.from_coords(zip(rup[0],
                                                  rup[1], rup[2])))
                            #
                            # update the list with the ruptures - the last
                            # element in the list is the container for the
                            # probability of occurrence. For the time being
                            # this is not defined
                            rups.append([srfc, wsum, dip, aspr, []])
                # update the list of ruptures
                lab = '{:.2f}'.format(mag)
                if lab in allrup:
                    allrup[lab] += rups
                else:
                    allrup[lab] = rups
            #
            # update counter
            iscnt += 1
    fh5.close()
    #
    #
    for lab in sorted(allrup.keys()):
        tmps = 'Number of ruptures for m={:s}: {:d}'
        logging.info(tmps.format(lab, len(allrup[lab])))
    #
    # compute the normalising factor for every rupture
    twei = {}
    for mag, occr in mfd.get_annual_occurrence_rates():
        smm = 0.
        lab = '{:.2f}'.format(mag)
        for _, wei, _, _, _ in allrup[lab]:
            if np.isfinite(wei):
                smm += wei
        twei[lab] = smm
        tmps = 'Total weight {:s}: {:f}'
        logging.info(tmps.format(lab, twei[lab]))
    #
    # generate and store the final set of ruptures
    fh5 = h5py.File(hdf5_filename, 'a')
    grp_rup = fh5.create_group('ruptures')
    #
    for mag, occr in mfd.get_annual_occurrence_rates():
        #
        # set label
        lab = '{:.2f}'.format(mag)
        #
        # warning
        if twei[lab] < 1e-50 and uniform_fraction < 0.99:
            tmps = 'Weight for magnitude {:s} equal to 0'
            tmps = tmps.format(lab)
            logging.warning(tmps)
        #
        #
        rups = []
        grp = grp_rup.create_group(lab)
        cnt = 0
        numrup = len(allrup[lab])
        for srfc, wei, _, _, _ in allrup[lab]:
            #
            # calculate the weight
            if twei[lab] > 1e-10:
                wei = wei / twei[lab]
                ocr = (wei * occr * (1.-uniform_fraction) +
                       occr / numrup * uniform_fraction)
            else:
                ocr = occr / numrup * uniform_fraction
            #
            # compute the probabilities
            p0 = np.exp(-ocr*tspan)
            p1 = 1. - p0
            #
            #
            rups.append([srfc, wei, dip, aspr, [p0, p1]])
            #
            #
            a = np.zeros(1, dtype=[('lons', 'f4', srfc.mesh.lons.shape),
                                   ('lats', 'f4', srfc.mesh.lons.shape),
                                   ('deps', 'f4', srfc.mesh.lons.shape),
                                   ('w', 'float32'),
                                   ('dip', 'f4'),
                                   ('aspr', 'f4'),
                                   ('prbs', 'float32', (2)),
                                   ])
            a['lons'] = srfc.mesh.lons
            a['lats'] = srfc.mesh.lats
            a['deps'] = srfc.mesh.depths
            a['w'] = wei
            a['dip'] = dip
            a['aspr'] = aspr
            a['prbs'] = np.array([p0, p1], dtype='float32')
            grp.create_dataset('{:08d}'.format(cnt), data=a)
            cnt += 1
        allrup[lab] = rups
    fh5.close()

    return allrup


def list_of_floats_from_string(istr):
    """
    """
    tstr = re.sub(r'(\[|\])', '', istr)
    return [float(d) for d in re.split(',', tstr)]


def dict_of_floats_from_string(istr):
    """
    """
    tstr = re.sub(r'(\{|\})', '', istr)
    out = {}
    for tmp in re.split('\,', tstr):
        elem = re.split(':', tmp)
        out[float(elem[0])] = float(elem[1])
    return out


def calculate_ruptures(ini_fname, ref_fdr=None):
    """
    :param str ini_fname:
        The name of a .ini file
    :param ref_fdr:
        The path to the reference folder used to set the paths in the .ini
        file. If not provided directly, we use the one set in the .ini file.
    """
    #
    # read config file
    config = configparser.ConfigParser()
    config.readfp(open(ini_fname))
    #
    # logging settings
    logging.basicConfig(format='rupture:%(levelname)s:%(message)s')
    #
    # reference folder
    if ref_fdr is None:
        ref_fdr = config.get('main', 'reference_folder')
    #
    # set parameters
    profile_sd_topsl = config.getfloat('main', 'profile_sd_topsl')
    edge_sd_topsl = config.getfloat('main', 'edge_sd_topsl')
    # this sampling distance is used to
    sampling = config.getfloat('main', 'sampling')
    float_strike = config.getfloat('main', 'float_strike')
    float_dip = config.getfloat('main', 'float_dip')
    slab_thickness = config.getfloat('main', 'slab_thickness')
    label = config.get('main', 'label')
    hspa = config.getfloat('main', 'hspa')
    vspa = config.getfloat('main', 'vspa')
    uniform_fraction = config.getfloat('main', 'uniform_fraction')
    #
    # MFD params
    agr = config.getfloat('main', 'agr')
    bgr = config.getfloat('main', 'bgr')
    mmax = config.getfloat('main', 'mmax')
    mmin = config.getfloat('main', 'mmin')
    #
    # set profile folder
    path = config.get('main', 'profile_folder')
    path = os.path.abspath(os.path.join(ref_fdr, path))
    #
    # catalogue
    cat_pickle_fname = config.get('main', 'catalogue_pickle_fname')
    cat_pickle_fname = os.path.abspath(os.path.join(ref_fdr, cat_pickle_fname))
    #
    # output
    hdf5_filename = config.get('main', 'out_hdf5_fname')
    hdf5_filename = os.path.abspath(os.path.join(ref_fdr, hdf5_filename))
    #
    # smoothing output
    out_hdf5_smoothing_fname = config.get('main', 'out_hdf5_smoothing_fname')
    tmps = os.path.join(ref_fdr, out_hdf5_smoothing_fname)
    out_hdf5_smoothing_fname = os.path.abspath(tmps)
    #
    # tectonic regionalisation
    treg_filename = config.get('main', 'treg_fname')
    if not re.search('[a-z]', treg_filename):
        treg_filename = None
    else:
        treg_filename = os.path.abspath(os.path.join(ref_fdr, treg_filename))
    #
    #
    dips = list_of_floats_from_string(config.get('main', 'dips'))
    asprsstr = config.get('main', 'aspect_ratios')
    asprs = dict_of_floats_from_string(asprsstr)
    #
    # magnitude-scaling relationship
    msrstr = config.get('main', 'mag_scaling_relation')
    msrd = get_available_scalerel()
    if msrstr not in msrd.keys():
        raise ValueError('')
    msr = msrd[msrstr]()
    #
    # ------------------------------------------------------------------------
    #
    print('Reading profiles from:', path)
    profiles, pro_fnames = _read_profiles(path)
    #
    """
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for ipro, (pro, fnme) in enumerate(zip(profiles, pro_fnames)):
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], 'x--b', markersize=2)
            tmps = '{:d}-{:s}'.format(ipro, os.path.basename(fnme))
            ax.text(tmp[0, 0], tmp[0, 1], tmp[0, 2], tmps)
        ax.invert_zaxis()
        ax.view_init(50, 55)
        plt.show()
    """
    #
    # create mesh from profiles
    msh = create_from_profiles(profiles, profile_sd=profile_sd_topsl,
                               edge_sd=edge_sd_topsl)
    #
    # Create inslab mesh. The one created here describes the top of the slab.
    # The output (i.e ohs) is a dictionary with the values of dip as keys. The
    # values in the dictionary are :class:`openquake.hazardlib.geo.line.Line`
    # instances
    ohs = create_inslab_meshes(msh, dips, slab_thickness, sampling)

    if 0:
        vsc = 0.1

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        #
        # profiles
        for ipro, (pro, fnme) in enumerate(zip(profiles, pro_fnames)):
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            tmp[tmp[:,0] < 0, 0] = tmp[tmp[:,0] < 0, 0] + 360
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2]*vsc, 'x--b', markersize=2)
            tmps = '{:d}-{:s}'.format(ipro, os.path.basename(fnme))
            ax.text(tmp[0, 0], tmp[0, 1], tmp[0, 2]*vsc, tmps)
        #
        # top of the slab mesh
        plot_mesh(ax, msh, vsc)
        #
        for key in ohs:
            for iii in range(len(ohs[key])):
                for line in ohs[key][iii]:
                    pnt = np.array([[p.longitude, p.latitude, p.depth]
                                    for p in line.points])
                    pnt[pnt[:,0] < 0, 0] = pnt[pnt[:,0] < 0, 0] + 360
                    ax.plot(pnt[:,0], pnt[:,1], pnt[:,2]*vsc, '-r')
        ax.invert_zaxis()
        ax.view_init(50, 55)
        plt.show()

    #
    # The one created here describes the bottom of the slab
    lmsh = create_lower_surface_mesh(msh, slab_thickness)
    #
    # get min and max values
    milo, mila, mide, malo, mala, made = get_min_max(msh, lmsh)
    #
    # discretizing the slab
    # omsh = Mesh(msh[:, :, 0], msh[:, :, 1], msh[:, :, 2])
    # olmsh = Mesh(lmsh[:, :, 0], lmsh[:, :, 1], lmsh[:, :, 2])
    #
    # this `dlt` value [in degrees] is used to create a buffer around the mesh
    dlt = 5.0
    msh3d = Grid3d(milo-dlt, mila-dlt, mide, malo+dlt, mala+dlt, made, hspa,
                   vspa)
    # mlo, mla, mde = msh3d.select_nodes_within_two_meshesa(omsh, olmsh)
    mlo, mla, mde = msh3d.get_coordinates_vectors()
    #
    # save data on hdf5 file
    if os.path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    fh5 = h5py.File(hdf5_filename, 'w')
    grp_slab = fh5.create_group('slab')
    grp_slab.create_dataset('top', data=msh)
    grp_slab.create_dataset('bot', data=lmsh)
    fh5.close()
    #
    # get catalogue
    catalogue = get_catalogue(cat_pickle_fname, treg_filename, label)
    #
    # smoothing - the output is
    values, smooth = smoothing(mlo, mla, mde, catalogue, hspa, vspa,
                               out_hdf5_smoothing_fname)
    #
    # spatial index
    # r = spatial_index(mlo, mla, mde, catalogue, hspa, vspa)
    r, proj = spatial_index(smooth)
    #
    # magnitude-frequency distribution
    mfd = TruncatedGRMFD(min_mag=mmin, max_mag=mmax, bin_width=0.1,
                         a_val=agr, b_val=bgr)
    # create all the ruptures - the probability of occurrence is for one year
    # in this case
    allrup = create_ruptures(mfd, dips, sampling, msr, asprs, float_strike,
                             float_dip, r, values, ohs, 1., hdf5_filename,
                             uniform_fraction, proj)


def main(argv):

    p = sap.Script(calculate_ruptures)
    p.arg(name='ini_fname', help='.ini filename')
    p.opt(name='ref_fdr', help='Reference folder for paths')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
