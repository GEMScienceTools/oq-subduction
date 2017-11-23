
import os
import glob
import numpy as np
import unittest
import shutil
import tempfile

from oq.hmtk.subduction.create_2pt5_model import (read_profiles_csv,
                                                  get_profiles_length,
                                                  write_profiles_csv,
                                                  write_edges_csv,
                                                  get_interpolated_profiles)

from openquake.hazardlib.geo.geodetic import distance


CS_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/cs')
CAM_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/cs_cam')


class WriteProfilesEdgesTest(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_write_profiles_edges(self):
        """
        Test writing edges
        """
        #
        # read data and compute distances
        sps, dmin, dmax = read_profiles_csv(CS_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        maximum_sampling_distance = 15
        num_sampl = np.ceil(lengths[longest_key] / maximum_sampling_distance)
        ssps = get_interpolated_profiles(sps, lengths, num_sampl)
        write_profiles_csv(ssps, self.test_dir)
        write_edges_csv(ssps, self.test_dir)
        #
        #
        tmp = []
        for fname in glob.glob(os.path.join(self.test_dir, 'cs*')):
            tmp.append(fname)
        self.assertEqual(len(tmp), 2)
        #
        #
        tmp = []
        for fname in glob.glob(os.path.join(self.test_dir, 'edge*')):
            tmp.append(fname)
        self.assertEqual(len(tmp), 7)


class ReadProfilesTest(unittest.TestCase):

    def test_read_profiles(self):
        """
        Test reading a profile file
        """
        sps, dmin, dmax = read_profiles_csv(CS_DATA_PATH)
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
        sps, dmin, dmax = read_profiles_csv(CS_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        # check shortest and longest profiles
        assert longest_key == '003'
        assert shortest_key == '004'
        # check lenghts
        expected = np.array([103.454865, 101.369319])
        computed = np.array([lengths[key] for key in sorted(sps.keys())])
        np.testing.assert_allclose(computed, expected, rtol=2)


class GetInterpolatedProfilesTest(unittest.TestCase):

    def test_interpolation_simple(self):
        """
        Test profile interpolation: simple case | sampling: 30 km
        """
        # read data and compute distances
        sps, dmin, dmax = read_profiles_csv(CS_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        maximum_sampling_distance = 30
        num_sampl = np.ceil(lengths[longest_key] / maximum_sampling_distance)
        #
        # get interpolated profiles
        ssps = get_interpolated_profiles(sps, lengths, num_sampl)
        lll = []
        for key in sorted(ssps.keys()):
            odat = sps[key]
            dat = ssps[key]
            distances = distance(dat[0:-2, 0], dat[0:-2, 1], dat[0:-2, 2],
                                 dat[1:-1, 0], dat[1:-1, 1], dat[1:-1, 2])
            expected = lengths[key] / num_sampl * np.ones_like(distances)
            np.testing.assert_allclose(distances, expected, rtol=3)
            #
            # update the list with the number of points in each profile
            lll.append(len(dat[:, 0]))
            #
            # check that the interpolated profile starts from the same point
            # of the original one
            self.assertListEqual([odat[0, 0], odat[0, 1]],
                                 [dat[0, 0], dat[0, 1]])
            # check that the depth of the profiles is always increasing
            computed = np.all(np.sign(np.diff(dat[:-1, 2]-dat[1:, 2])) < 0)
            self.assertTrue(computed)
        #
        # check that all the profiles have all the same length
        dff = np.diff(np.array(lll))
        zeros = np.zeros_like(dff)
        np.testing.assert_allclose(dff, zeros, rtol=2)

    def test_interpolation_cam(self):
        """
        Test profile interpolation: CAM | maximum sampling: 30 km
        """
        #
        # read data and compute distances
        sps, dmin, dmax = read_profiles_csv(CAM_DATA_PATH)
        lengths, longest_key, shortest_key = get_profiles_length(sps)
        maximum_sampling_distance = 30.
        num_sampl = np.ceil(lengths[longest_key] / maximum_sampling_distance)
        #
        # get interpolated profiles
        ssps = get_interpolated_profiles(sps, lengths, num_sampl)
        lll = []
        for key in sorted(ssps.keys()):
            odat = sps[key]
            dat = ssps[key]
            print(dat)

            distances = distance(dat[0:-2, 0], dat[0:-2, 1], dat[0:-2, 2],
                                 dat[1:-1, 0], dat[1:-1, 1], dat[1:-1, 2])
            expected = lengths[key] / num_sampl * np.ones_like(distances)
            np.testing.assert_allclose(distances, expected, rtol=3)
            #
            # update the list with the number of points in each profile
            lll.append(len(dat[:, 0]))
            #
            # check that the interpolated profile starts from the same point
            # of the original one
            self.assertListEqual([odat[0, 0], odat[0, 1]],
                                 [dat[0, 0], dat[0, 1]])
            #
            # check that the depth of the profiles is always increasing
            print(dat[:-1, 2])
            print(dat[1:, 2])
            print('dff', dat[:-1, 2]-dat[1:, 2])
            computed = np.all(np.sign(dat[:-1, 2]-dat[1:, 2]) < 0)
            print(computed)
            self.assertTrue(computed)
        #
        # check that all the profiles have all the same length
        dff = np.diff(np.array(lll))
        zeros = np.zeros_like(dff)
        np.testing.assert_allclose(dff, zeros, rtol=2)
