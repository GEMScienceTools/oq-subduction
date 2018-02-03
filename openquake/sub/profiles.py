#!/usr/bin/env python

import os
import re
import glob
import numpy as np

from pathlib import Path
from scipy.interpolate import CubicSpline
from openquake.hazardlib.geo import Line, Point


def _from_lines_to_array(lines):
    """
    :param lines:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    :returns:
        A 2D :class:`numpy.ndarray` instance with 3 columns and as many rows
        as the number of points composing all the lines
    """
    out = []
    for line in lines:
        for pnt in line:
            x = float(pnt.longitude)
            y = float(pnt.latitude)
            z = float(pnt.depth)
        out.append([x, y, z])
    return np.array(out)


def _from_line_to_array(line):
    """
    :param list line:
        A :class:`openquake.hazardlib.geo.line.Line` instance
    :returns:
        A 2D :class:`numpy.ndarray` instance with 3 columns and as many rows
        as the number of points composing the line
    """
    out = np.array((len(line.points, 3)))
    for i, pnt in enumerate(line.points):
        out[:, 0] = float(pnt.longitude)
        out[:, 1] = float(pnt.latitude)
        out[:, 2] = float(pnt.depth)
    return out


class ProfileSet():
    """
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """

    def __init__(self, profiles=[]):
        self.profiles = profiles

    @classmethod
    def from_files(cls, fname):
        """
        """
        lines = []
        for filename in sorted(glob.glob(os.path.join(fname, 'cs*.csv'))):
            tmp = np.loadtxt(filename)
            pnts = []
            for i in range(tmp.shape[0]):
                pnts.append(Point(tmp[i, 0], tmp[i, 1], tmp[i, 2]))
            lines.append(Line(pnts))
            #
            # Profile ID
            fname = Path(os.path.basename(filename)).stem
            sid = re.sub('^cs_', '', fname)
            sid = '%03d' % int(sid)
        return cls(lines)

    def get_cubic_spline(self):
        """
        """
        arr = _from_lines_to_array(self.profiles)
        return CubicSpline(arr[:, 2], arr[:, 0:1])
