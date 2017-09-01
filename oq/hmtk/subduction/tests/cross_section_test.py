# uncompyle6 version 2.11.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.5.3 (default, Apr 23 2017, 18:09:27) 
# [GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.42.1)]
# Embedded file name: /Users/mpagani/Projects/hmtk/hmtk/subduction/cross_section_test.py
# Compiled at: 2016-08-08 15:46:36
import numpy
import unittest
from hmtk.seismicity.catalogue import Catalogue
from hmtk.subduction.cross_sections import CrossSection
from hmtk.subduction.cross_sections import get_min_distance
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.line import Line

class CrossSectionTest(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.csect = CrossSection(10.0, 45.0, [100], [45])
        data = numpy.array([[1900, 10.0, 45.0, 5.0],
         [
          1901, 10.1, 45.1, 5.1],
         [
          1902, 10.11, 45.12, 5.2],
         [
          1903, 10.22, 45.21, 5.3],
         [
          1904, 10.31, 45.32, 5.4]])
        keys = ['year', 'longitude', 'latitude', 'magnitude']
        self.ctlg = Catalogue()
        self.ctlg.load_from_array(keys, data)

    def test_get_eqks_within_buffer(self):
        """
        """
        pass


class MinDistTest(unittest.TestCase):

    def setUp(self):
        pass

    def test01(self):
        lons = [
         10.0, 11.0, 12.0]
        lats = [45.0, 44.0, 43.0]
        lons = [10.0, 11.0]
        lats = [45.0, 46.0]
        pnts = numpy.array([[10.3, 45.4], [9.8, 45.4],
         [
          10.3, 44.8], [9.8, 44.8]])
        pnts = numpy.array([[10.3, 45.0], [9.0, 45.0]])
        line = Line([ Point(lo, la) for lo, la in zip(lons, lats) ])
        get_min_distance(line, pnts)
        assert 0 == 1
# okay decompiling cross_section_test.pyc
