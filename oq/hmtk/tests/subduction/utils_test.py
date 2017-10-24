import unittest
import numpy

from oq.hmtk.subduction.utils import (get_line_of_intersection,
                                      get_direction_cosines)


class TestGetDirectionCosines(unittest.TestCase):

    def test_vertical_fault01(self):
        strike = 0.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([1.0, 0, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_fault02(self):
        strike = 45.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.7071068, -0.7071068, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_fault03(self):
        strike = 225.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([-0.7071068, +0.7071068, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault01(self):
        strike = 0.0
        dip = 45.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.7071068, 0.0, 0.7071068])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault02(self):
        strike = -45.0
        dip = 45.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.5, 0.5, 0.7071068])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault03(self):
        strike = 210
        dip = 30.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([-0.4330127, 0.25,  0.86602540])
        numpy.testing.assert_almost_equal(actual, desired)


class TestLineOfPlaneIntersection(unittest.TestCase):

    def test_vertical_faults01(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 90.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, 0.0, -1.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_faults02(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 315.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, 0.0, 1.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane01(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 0.0
        dip2 = 45.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, -1.0, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane02(self):
        strike1 = 45.0
        dip1 = 45.0
        strike2 = 225.0
        dip2 = 45.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    """
    def test_dipping_plane03(self):
        strike1 = 140.
        dip1 = 50.0
        strike2 = 30.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane04(self):
        strike1 = 60.
        dip1 = 32.0
        strike2 = 30.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)
    """
