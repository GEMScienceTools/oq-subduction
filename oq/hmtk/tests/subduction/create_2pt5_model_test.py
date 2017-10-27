
import os
import numpy as np
import unittest

from oq.hmtk.subduction.create_2pt5_model import (read_profiles_csv,
                                                  get_profiles_length,
                                                  get_interpolated_profiles)

from openquake.hazardlib.geo.geodetic import distance


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/cs')


class ReadProfilesTest(unittest.TestCase):

    def test_read_profiles(self):
        """
        Test reading a profile file
        """
        sps, dmin, dmax = read_profiles_csv(BASE_DATA_PATH)
        # check the minimum and maximum depths computed
        assert dmin == 0
        assert dmax == 40.0
        expected_keys = ['003', '004']
        # check the keys
        self.assertListEqual(expected_keys, sorted(list(sps.keys())))
        # check the coordinates of the profile
        expected = np.array([[10., 45., 0.],
                             [10.2, 45.2, 10.],
                             [10.3, 45.3, 15.],
                             [10.5, 45.5, 25.],
                             [10.7, 45.7, 40.]])
        np.testing.assert_allclose(sps['003'], expected, rtol=2)


class GetProfilesLengthTest(unittest.TestCase):

    def test_length_calc_01(self):
        """
        Test computing the lenght of profiles
        """
        # read data and compute distances
        sps, dmin, dmax = read_profiles_csv(BASE_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        # check shortest and longest profiles
        assert longest_key == '003'
        assert shortest_key == '004'
        # check lenghts
        expected = np.array([103.454865, 101.369319])
        computed = np.array([lengths[key] for key in sorted(sps.keys())])
        np.testing.assert_allclose(computed, expected, rtol=2)


class GetInterpolatedProfilesTest(unittest.TestCase):

    def test_interpolation(self):
        """
        Test profile interpolation
        """
        # read data and compute distances
        sps, dmin, dmax = read_profiles_csv(BASE_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        maximum_sampling_distance = 15
        num_sampl = np.ceil(lengths[longest_key] / maximum_sampling_distance)

        ssps = get_interpolated_profiles(sps, lengths, num_sampl)
        for key in ssps.keys():
            dat = ssps[key]
            distances = distance(dat[0:-2, 0], dat[0:-2, 1], dat[0:-2, 2],
                                 dat[1:-1, 0], dat[1:-1, 1], dat[1:-1, 2])
            expected = lengths[key] / num_sampl * np.ones_like(distances)
            np.testing.assert_allclose(distances, expected, rtol=2)
