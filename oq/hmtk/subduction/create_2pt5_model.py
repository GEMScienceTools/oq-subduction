#!/usr/bin/env python

import os
import re
import sys
import glob
import numpy


from openquake.hazardlib.geo.geodetic import distance, point_at, azimuth


def get_profiles_length(sps):
    """
    :parameter dict sps:
        A dictionary containing the subduction profiles
    :returns:
        A dictionary where key is the ID of the profile and value is the length
        and, a string identifying the longest profile
    """
    lengths = {}
    longest_key = None
    shortest_key = None
    longest_length = 0.
    shortest_length = 1e10
    for key in sorted(sps.keys()):
        dat = sps[key]
        total_length = 0
        for idx in range(0, len(dat)-1):
            dst = distance(dat[idx,0], dat[idx,1], dat[idx,2],
                           dat[idx+1,0], dat[idx+1,1], dat[idx+1,2])
            total_length += dst
        lengths[key] = total_length
        if longest_length < total_length:
            longest_length = total_length
            longest_key = key
        if shortest_length > total_length:
            shortest_length = total_length
            shortest_key = key
    return lengths, longest_key, shortest_key


def get_interpolated_profiles(sps, lengths, number_of_samples):
    """
    :parameter dict sps:
        A dictionary containing the subduction profiles key is a string and
        value is an instance of :class:`numpy.ndarray`
    :parameter dict lengths:
        A dictionary containing the subduction profiles lengths
    :parameter float number_of_samples:
        Number of subsegments to be created
    """
    ssps = {}
    for key in sorted(sps.keys()):
        # Sampling distance
        samp = lengths[key] / number_of_samples * 0.9999
        print('samp:', samp)
        # Data
        dat = sps[key]
        # Azimuth of the subduction profile
        azim = azimuth(dat[0, 0], dat[0, 1], dat[-1, 0], dat[-1, 1])
        # Initialise parameters
        idx = 0
        tdst = 0.0
        thdst = 0.0
        spro = [[dat[0, 0], dat[0, 1], dat[0, 2]]]
        # Process all the segments composing the profile
        while idx < len(dat)-1:
            #
            # segment length
            dst = distance(dat[idx, 0], dat[idx, 1], dat[idx, 2],
                           dat[idx+1, 0], dat[idx+1, 1], dat[idx+1, 2])
            #
            # segment dip angle
            dipr = numpy.arcsin((dat[idx+1, 2]-dat[idx, 2])/dst)
            #
            # segment horizontal distance
            fact = numpy.cos(dipr)
            hdst = dst*fact
            #
            # We take a sample if the lenght available (i.e. tdst+dst) is
            # larger than the sampling distance
            if tdst+dst > samp:
                #
                # number of subsegments fitting in this segment
                npoints = numpy.floor((tdst+dst) / samp)
                #
                # horizontal distance between the first point of the segment
                # and each new sampled points
                dsts = numpy.arange(1, npoints+1)*(samp*fact)-(tdst*fact)
                #
                # longitude and latitude of the new points
                nlo, nla = point_at(dat[idx, 0], dat[idx, 1], azim, dsts)
                #
                # depths
                nde = dat[idx, 2]+(dat[idx+1, 2]-dat[idx, 2])/hdst*dsts
                #
                # checking:
                # - all the interpolated depths must be within the upper
                #   and lower limit
                assert numpy.all(nde >= dat[idx, 2])
                assert numpy.all(nde <= dat[idx+1, 2])
                #
                # checking:
                # -
                tmp = (thdst+dsts)/(samp*fact)
                assert numpy.all(abs(tmp-numpy.arange(1, len(tmp)+1)) <= 2e-1)
                #
                # store results
                for lo, la, de in zip(nlo, nla, nde):
                    spro.append([lo, la, de])
                #
                # distance between the shallowest point of this segment and
                # the deepest sampled point
                tmp = distance(dat[idx, 0], dat[idx, 1], dat[idx, 2],
                               nlo[-1], nla[-1], nde[-1])
                tdst = dst - tmp
                thdst = hdst - dsts[-1]
                assert abs(thdst - tdst*fact) < 1e-1
            else:
                tdst += dst
                thdst += hdst
            idx += 1
        # Saving results
        if len(spro):
            ssps[key] = numpy.array(spro)
        else:
            print('length = 0')
    return ssps


def read_profiles_csv(foldername, upper_depth=0, lower_depth=1000,
                      from_id=".*", to_id="-999999"):
    """
    :parameter str foldername:
        The name of the folder containing the set of digitized profiles
    """
    dmin = +1e100
    dmax = -1e100
    sps = {}
    #
    # reading files
    read_file = False
    for filename in glob.glob(os.path.join(foldername, 'cs*.csv')):
        #
        # check the file key
        if re.search(from_id, filename):
            read_file = True
        elif re.search(to_id, filename):
            read_file = False

        if read_file:
            sid = re.sub('^cs_', '', re.split('\.',
                                              os.path.basename(filename))[0])
            if not re.search('[a-zA-Z]', sid):
                sid = '%03d' % int(sid)
            tmpa = numpy.loadtxt(filename)
            #
            # selecting depths within the defined range
            j = numpy.nonzero((tmpa[:, 2] >= upper_depth) &
                              (tmpa[:, 2] <= lower_depth))
            if len(j[0]):
                sps[sid] = tmpa[j[0], :]
                dmin = min(min(sps[sid][:, 2]), dmin)
                dmax = max(max(sps[sid][:, 2]), dmax)
    return sps, dmin, dmax


def write_profiles_csv(sps, foldername):
    """
    :parameter dic sps:
    :parameter str foldername:
        The name of the file which contains the interpolated profiles
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    for key in sorted(sps):
        dat = numpy.array(sps[key])
        fname = os.path.join(foldername, 'cs_%s.csv' % (key))
        numpy.savetxt(fname, dat)


def write_edges_csv(sps, foldername):
    """
    :parameter dic sps:
    :parameter str foldername:
        The name of the file which contains the interpolated profiles
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    for idx in range(0, len(sps[list(sps.keys())[0]])):
        dat = []
        for key in sorted(sps):
            dat.append(sps[key][idx, :])
        fname = os.path.join(foldername, 'edge_%03d.csv' % (idx))
        numpy.savetxt(fname, numpy.array(dat))


def main(argv):
    """
    argv[0] - Folder name
    argv[1] - Sampling distance [km]
    argv[2] - Output folder name
    argv[3] - Maximum sampling distance
    """
    in_path = os.path.abspath(argv[0])
    out_path = os.path.abspath(argv[2])
    #
    # Check input
    if len(argv) < 3:
        tmps = 'Usage: create_subduction_model.py <in_folder>'
        tmps += ' <ini_filename> <out_folder>'
        print(tmps)
        exit(0)
    #
    # Sampling distance [km]
    if len(argv) < 4:
        maximum_sampling_distance = 25.
    else:
        maximum_sampling_distance = float(argv[3])
    #
    # Check folders
    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input: {0:s}\n'.format(in_path)
        tmps += '    input: {0:s}\n'.format(out_path)
        print(tmps)
        exit(0)
    #
    # Read profiles
    sps, dmin, dmax = read_profiles_csv(in_path)
    #
    # Compute lengths
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    number_of_samples = numpy.ceil(lengths[longest_key] /
                                   maximum_sampling_distance)
    print('Number of subsegments:', number_of_samples)
    tmp = lengths[shortest_key]/number_of_samples
    print('Shortest sampling [%s]: %.4f' % (shortest_key, tmp))
    tmp = lengths[longest_key]/number_of_samples
    print('Longest sampling  [%s]: %.4f' % (longest_key, tmp))
    #
    # Resampled profiles
    rsps = get_interpolated_profiles(sps, lengths, number_of_samples)
    #
    # Store profiles
    write_profiles_csv(rsps, out_path)
    #
    # Store edges
    write_edges_csv(rsps, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
