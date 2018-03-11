"""
"""

import os
import unittest

from openquake.sub.slab.rupture import calculate_ruptures
from openquake.sub.build_complex_surface import build_complex_surface

BASE_DATA_PATH = os.path.dirname(__file__)


class RuptureSetCreationTest(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        relpath = '../data/ini/test.ini'
        self.ini_fname = os.path.join(BASE_DATA_PATH, relpath)
        #
        # prepare the input folder and the output folder
        in_path = os.path.join(BASE_DATA_PATH, './../data/sp_cam/')
        out_path = os.path.join(BASE_DATA_PATH, './../data/tmp/')
        #
        # first we create the complex surface. We use the profiles used for
        # the subduction in CCARA
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, out_path,
                              upper_depth=50, lower_depth=200)

    def test_01(self):
        """
        Test rupture creation
        """
        reff = os.path.join(BASE_DATA_PATH, './../data/ini/')
        calculate_ruptures(self.ini_fname, reff)
