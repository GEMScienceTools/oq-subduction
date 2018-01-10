#!/usr/bin/env python

import os
import re
import sys
import glob
import h5py
import numpy
import filecmp
import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openquake.hazardlib.geo.geodetic import distance, point_at, azimuth

from oq.hmtk.subduction.create_2pt5_model import (read_profiles_csv,
                                                  get_profiles_length,
                                                  get_interpolated_profiles,
                                                  write_edges_csv,
                                                  write_profiles_csv)


def main(argv):
    """
    argv[0] - Folder name
    argv[1] - Sampling distance [km]
    argv[2] - Output folder name
    argv[3] - Upper seismogenic depth [km]
    argv[4] - Lower seismogenic depth [km]
    """
    #
    #
    in_path = os.path.abspath(argv[0])
    maximum_sampling_distance = float(argv[1])
    out_path = os.path.abspath(argv[2])
    #
    # set upper depth
    upper_depth = 0.0
    if len(argv) > 3:
        upper_depth = float(argv[3])
    #
    # set lower depth
    lower_depth = 1000.
    if len(argv) > 4:
        lower_depth = float(argv[4])
    #
    # selection label
    from_id = ".*"
    if len(argv) > 5:
        from_id = argv[5]
    #
    # selection label
    to_id = ".*"
    if len(argv) > 6:
        to_id = argv[6]
    #
    # Check input
    if len(argv) < 3:
        tmps = 'Usage: build_complex_surface.py <in_folder>'
        tmps += ' <sampling_dist> <out_folder> <lower_seism_depth>'
        print (tmps)
        exit(0)
    #
    # Check input and output folders
    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input: {0:s}\n'.format(in_path)
        tmps += '    input: {0:s}\n'.format(out_path)
        print (tmps)
        exit(0)
    #
    # read profiles
    sps, dmin, dmax = read_profiles_csv(in_path, upper_depth, lower_depth,
                                        from_id, to_id)

    #
    # compute length of profiles
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    print('key:', longest_key, shortest_key)
    #
    #
    number_of_samples = numpy.ceil(lengths[longest_key] /
                                   maximum_sampling_distance)
    print ('Number of subsegments:', number_of_samples)
    tmp = lengths[shortest_key]/number_of_samples
    print ('Shortest sampling [%s]: %.4f' % (shortest_key, tmp))
    tmp = lengths[longest_key]/number_of_samples
    print ('Longest sampling  [%s]: %.4f' % (longest_key, tmp))
    #
    # resampled profiles
    rsps = get_interpolated_profiles(sps, lengths, number_of_samples)
    #
    # store new profiles
    write_profiles_csv(rsps, argv[2])
    #
    # store computed edges
    write_edges_csv(rsps, argv[2])

if __name__ == "__main__":
    main(sys.argv[1:])
