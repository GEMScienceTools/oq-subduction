#!/usr/bin/env python
import os
import re
import sys
import glob
import numpy

from pyproj import Proj
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
            dst = distance(dat[idx, 0], dat[idx, 1], dat[idx, 2],
                           dat[idx+1, 0], dat[idx+1, 1], dat[idx+1, 2])
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
        #
        # calculate the sampling distance
        samp = lengths[key] / number_of_samples
        #print (samp)
        #
        # set data for the profile
        dat = sps[key]
        #
        # initialise
        idx = 0
        cdst = 0
        spro = [[dat[0, 0], dat[0, 1], dat[0, 2]]]
        #
        # process the segments composing the profile
        while idx < len(dat)-1:
            #
            # segment length and azimuth
            dst = distance(dat[idx,0],dat[idx,1],dat[idx,2],dat[idx+1,0],dat[idx+1,1],dat[idx+1,2])
            azim = azimuth(dat[idx,0],dat[idx,1],dat[idx+1,0],dat[idx+1,1])
            #
            # calculate total distance i.e. cumulated + new segment
            total_dst = cdst + dst
            #
            # number of new points
            num_new_points = int(numpy.floor(total_dst/samp))
            #
            # take samples if possible
            if num_new_points > 0:
                #print ('new points!',num_new_points)
                #
                # segment dip angle 
                dipr = numpy.arcsin((dat[idx+1, 2]-dat[idx, 2])/dst)
                hfact = numpy.cos(dipr)
                vfact = numpy.sin(dipr)
                #
                for i in range(0, num_new_points):
                    tdst = (i+1) * samp - cdst
                    hdst = tdst * hfact
                    tlo,tla = point_at(dat[idx,0],dat[idx,1],azim,hdst)
                    vdst = tdst * vfact
                    spro.append([tlo, tla, dat[idx, 2]+vdst])
                    #
                    # check distance with the previous point
                    if i > 0:
                        check = distance(tlo, tla, dat[idx, 2]+vdst,
                                         spro[-2][0], spro[-2][1], spro[-2][2])
                        if abs(check - samp) > samp*0.15:
                            msg = 'Distance between consecutive points'
                            msg += ' is incorrect: {:.3f} {:.3f}'.format(check,
                                                                         samp)
                            raise ValueError(msg)
                #
                # new distance left over
                cdst = (dst + cdst) - num_new_points * samp
            else:
                cdst += dst
            #
            # updating index
            idx += 1
        #
        # Saving results
        if len(spro):
            ssps[key] = numpy.array(spro)
        else:
            print('length = 0')
    return ssps


def get_interpolated_profiles_old(sps, lengths, number_of_samples):
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
        #
        # calculate the sampling distance
        samp = lengths[key] / number_of_samples
        #
        # set data for the profile
        dat = sps[key]
        #
        # projecting profile coordinates
        p = Proj('+proj=lcc +lon_0={:f}'.format(dat[0, 0]))
        x, y = p(dat[:, 0], dat[:, 1])
        x = x / 1e3  # m -> km
        y = y / 1e3  # m -> km

        #
        # horizontal 'slope'
        hslope = numpy.arctan((y[-1]-y[0]) / (x[-1]-x[0]))
        xfact = numpy.cos(hslope)
        yfact = numpy.sin(hslope)
        #
        # initialise
        idx = 0
        cdst = 0
        spro = [[dat[0, 0], dat[0, 1], dat[0, 2]]]
        #
        # process the segments composing the profile
        while idx < len(dat)-1:
            #
            # segment length
            dst = ((x[idx] - x[idx+1])**2 + (y[idx] - y[idx+1])**2 +
                   (dat[idx, 2] - dat[idx+1, 2])**2)**.5
            #
            # calculate total distance i.e. cumulated + new segment
            total_dst = cdst + dst
            #
            # number of new points
            num_new_points = int(numpy.floor(total_dst/samp))
            #
            # take samples if possible
            if num_new_points > 0:
                #
                # segment dip angle
                dipr = numpy.arcsin((dat[idx+1, 2]-dat[idx, 2])/dst)
                hfact = numpy.cos(dipr)
                vfact = numpy.sin(dipr)
                ##
                #
                for i in range(0, num_new_points):
                    tdst = (i+1) * samp - cdst
                    hdst = tdst * hfact
                    vdst = tdst * vfact
                    tlo, tla = p((x[idx] + hdst*xfact)*1e3,
                                 (y[idx] + hdst*yfact)*1e3, inverse=True)
                    spro.append([tlo, tla, dat[idx, 2]+vdst])
                    #
                    # check distance with the previous point
                    if i > 0:
                        check = distance(tlo, tla, dat[idx, 2]+vdst,
                                         spro[-2][0], spro[-2][1], spro[-2][2])
                        if abs(check - samp) > samp*0.15:
                            msg = 'Distance between consecutive points'
                            msg += ' is incorrect: {:.3f} {:.3f}'.format(check,
                                                                         samp)
                            raise ValueError(msg)
                #
                # new distance left over
                cdst = (dst + cdst) - num_new_points * samp
            else:
                cdst += dst
            #
            # updating index
            idx += 1
        #
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
        A dictionary where keys are the profile labels and values are
        :class:`numpy.ndarray` instances
    :parameter str foldername:
        The name of the file which contains the interpolated profiles
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    #
    # run for all the edges i.e. number of
    max_num = len(sps[list(sps.keys())[0]])
    for idx in range(0, max_num-1):
        dat = []
        for key in sorted(sps):
            dat.append(sps[key][idx, :])
        fname = os.path.join(foldername, 'edge_%03d.csv' % (idx))
        numpy.savetxt(fname, numpy.array(dat))


def main(argv):
    """
    #old doc
    argv[0] - Folder name
    argv[1] - Sampling distance [km] # erasing so increments change!!!!
    argv[2] - Output folder name
    argv[3] - Maximum sampling distance # am now computing this from segments
    #new doc
    argv[0] - input folder name
    argv[1] - output folder name
    argv[2] - sampling distance
    """
    #with new setup
    in_path = os.path.abspath(argv[0])
    out_path = os.path.abspath(argv[1])
    #
    # Check input
    if len(argv) < 2:
        tmps = 'Usage: create_subduction_model.py <in_folder>'
        tmps += ' <out_folder>'
        print(tmps)
        exit(0)
    #
    
     #Sampling distance [km] SHOULD WE COMPUTE THIS INSTEAD? 
    if len(argv) < 3:
        maximum_sampling_distance = 25.
    else:
        maximum_sampling_distance = float(argv[2])
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
    # Compute lengths -> note: max_sampling_distance is actually a SMALL number 
    lengths, longest_key, shortest_key = get_profiles_length(sps)
    number_of_samples = numpy.ceil(lengths[longest_key]/maximum_sampling_distance)
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
