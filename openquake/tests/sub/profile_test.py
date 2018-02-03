
import os
import numpy as np
import unittest

from oq.hmtk.subduction.profiles import ProfileSet
from openquake.hazardlib.geo.geodetic import distance


class ProfileTest(unittest.TestCase):
    """
    """

    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

    def setUp(self):
        #
        # trench filename
        self.dname_profile = os.path.join(self.BASE_DATA_PATH, 'cs_cam')

    def test_reading_folder(self):
        """
        Read profiles from a folder
        """
        prfs = ProfileSet.from_files(self.dname_profile)
        self.assertEqual(27, len(prfs.profiles))
        assert 0 == 1

    def test_cubic_spline(self):
        """
        Create and use a cubic spline interpolator
        """
