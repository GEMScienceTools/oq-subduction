import scipy
import numpy
import matplotlib.pyplot as plt

from openquake.hazardlib.scalerel.wc1994 import WC1994


def mecclass(plungt, plungb, plungp):
    """
    This is taken from the FMC package.
    See https://josealvarezgomez.wordpress.com/

    It provides a classification of the rupture mechanism based on the
    Kaverina et al. (1996) methodology.

    :parameter plungt:
    :parameter plungb:
    :parameter plungp:

    """
    plunges = numpy.asarray((plungp, plungb, plungt))
    P = plunges[0]
    B = plunges[1]
    T = plunges[2]
    maxplung, axis = plunges.max(0), plunges.argmax(0)
    if maxplung >= 67.5:
            if axis == 0:  # P max
                    clase = 'N'  # normal faulting
            elif axis == 1:  # B max
                    clase = 'SS'  # strike-slip faulting
            elif axis == 2:  # T max
                    clase = 'R'  # reverse faulting
    else:
            if axis == 0:  # P max
                    if B > T:
                            clase = 'N-SS'  # normal - strike-slip faulting
                    else:
                            clase = 'N'  # normal faulting
            if axis == 1:  # B max
                    if P > T:
                            clase = 'SS-N'  # strike-slip - normal faulting
                    else:
                            clase = 'SS-R'  # strike-slip - reverse faulting
            if axis == 2:  # T max
                    if B > P:
                            clase = 'R-SS'  # reverse - strike-slip faulting
                    else:
                            clase = 'R'  # reverse faulting
    return clase


def get_direction_cosines(strike, dip):
    """
    Compute the direction cosines of the plane defined by the strike-dip tuple.

    :parameter strike:
        Strike of the plane. Defined using the right hand rule
    :parameter dip:
        Dip of the plane. Defined using the right hand rule
    :return:
        A 3x1 array containing the direction cosines of the normal to the plane
    """
    if dip < 89.99:
        c = scipy.cos(scipy.radians(dip))
        h = scipy.sin(scipy.radians(dip))
    else:
        c = 0.
        h = 1.
    a = h*scipy.sin(scipy.radians(strike+90.))
    b = h*scipy.cos(scipy.radians(strike+90.))
    den = numpy.sqrt(a**2.+b**2.+c**2.)
    a /= den
    b /= den
    c /= den
    return a, b, c


def plane_intersection(pln1, pln2):
    """
    Given two planes defined in the Hessian form
    (see http://mathworld.wolfram.com/HessianNormalForm.html) each one
    represented by 4x1 numpy array (nx, ny, nz, p) compute the line formed
    by the intersection between the two planes.

    :parameter pln1:
        A 4x1 array with direction cosines of the first plane
    :parameter pln2:
        A 4x1 array with direction cosines of the second plane
    :return:
        An array with the direction cosines of the line
    """
    dirc = numpy.cross(pln1[:-1], pln2[:-1])
    nrm = (sum(dirc**2))**.5
    return dirc / nrm


def get_line_of_intersection(strike1, dip1, strike2, dip2):
    """
    Find the direction cosines of the line obtained by the intersection between
    two planes defined in terms of strike and dip.

    :parameter strike1:
    :parameter dip1:
    :parameter strike2:
    :parameter dip2:
    """
    a, b, c = get_direction_cosines(strike1, dip1)
    acs, bcs, ccs = get_direction_cosines(strike2, dip2)
    pln1 = numpy.array([a, b, c, 0])
    pln2 = numpy.array([acs, bcs, ccs, 0])
    # inter contains the direction cosines of the line obtained by the
    # intersection between the two planes
    return plane_intersection(pln1, pln2)


def plot_planes_at(x, y, strikes, dips, magnitudes, strike_cs, dip_cs,
                   aratio=1.0, msr=None, ax=None, zorder=20, color='None',
                   linewidth=1):
    """
    This plots an a cross-section a number of rupture planes defined in terms
    of a strike and a dip.

    :parameter x:
        Coordinates x on the cross-section
    :parameter y:
        Coordinates y on the cross-section (it corresponds to a depth)
    :parameter strikes:
        Strike values of the planes
    :parameter dips:
        Dip values of the planes
    :parameter strike_cs:
        Strike angle of the cross-section plane [in degrees]
    :parameter dip_cs:
        Dip angle of the cross-section plane [in degrees]
    """
    if msr is None:
        msr = WC1994()
    cols = ['red', 'blue', 'green']
    for strike, dip, col, mag in zip(strikes, dips, cols, magnitudes):

        area = msr.get_median_area(mag, None)
        width = (area / aratio)**.5
        length = width * aratio
        t = numpy.arange(-width/2, width/2, 0.1)

        inter = get_line_of_intersection(strike, dip, strike_cs, dip_cs)
        xl = t*inter[0]
        yl = t*inter[1]
        zl = t*inter[2]
        ds = -numpy.sign(t)*(xl**2+yl**2)**.5 + x
        plt.axis('equal')

        if color is not None:
            col = color

        plt.plot(ds, zl+y, zorder=zorder, color=col, linewidth=linewidth)
