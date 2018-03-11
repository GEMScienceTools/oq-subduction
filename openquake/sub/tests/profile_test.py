
import os
import unittest

from openquake.sub.profiles import ProfileSet


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
