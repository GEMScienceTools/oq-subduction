import re
import numpy
import scipy
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT
from obspy.imaging.beachball import beach

from netCDF4 import Dataset

from hmtk.seismicity.selector import CatalogueSelector
from hmtk.subduction.cross_sections import CrossSection

from openquake.hazardlib.geo.geodetic import geodetic_distance

from utils import plot_planes_at, mecclass


def get_cmt(cmt_cat, cross_sect, lomin, lomax, lamin, lamax):
    loc = cmt_cat.data['longitude']
    lac = cmt_cat.data['latitude']
    idxs = cross_sect.get_grd_nodes_within_buffer(loc, lac, 50.,
                                                  lomin, lomax, lamin, lamax)
    return idxs


def get_crust1pt0_moho_depth(cross_sect, minlo=-180, maxlo=180, minla=-90,
                             maxla=90):
    filename = "/Users/mpagani/Google Drive/GEM_hazard/Data/Geology/CRUST1.0/depthtomoho.xyz"
    datal = []
    for line in open(filename, 'r'):
        xx = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
        datal.append([float(val) for val in xx])
    dataa = numpy.array(datal)
    idxs = cross_sect.get_grd_nodes_within_buffer(dataa[:, 0],
                                                  dataa[:, 1], 100.,
                                                  minlo, maxlo, minla, maxla)
    if idxs is not None and len(idxs):
        boo = numpy.zeros_like(dataa[:, 0], dtype=int)
        boo[idxs[0]] = 1
        return dataa[idxs, :]
    else:
        return None


def findSubsetIndices(min_lat, max_lat, min_lon, max_lon, lats, lons):
    """
    Array to store the results returned from the function. This is taken
    from http://www.trondkristiansen.com/?page_id=846
    """
    res = numpy.zeros((4), dtype=numpy.float64)
    minLon = min_lon
    maxLon = max_lon
    distances1 = []
    distances2 = []
    indices = []
    index = 1

    for point in lats:
        s1 = max_lat-point  # (vector subtract)
        s2 = min_lat-point  # (vector subtract)
        distances1.append((numpy.dot(s1, s1), point, index))
        distances2.append((numpy.dot(s2, s2), point, index-1))
        index = index+1
    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    distances1 = []
    distances2 = []
    index = 1

    for point in lons:
        s1 = maxLon-point  # (vector subtract)
        s2 = minLon-point  # (vector subtract)
        distances1.append((numpy.dot(s1, s1), point, index))
        distances2.append((numpy.dot(s2, s2), point, index-1))
        index = index+1

    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    minJ = indices[1][2]
    maxJ = indices[0][2]
    minI = indices[3][2]
    maxI = indices[2][2]

    res[0] = minI
    res[1] = maxI
    res[2] = minJ
    res[3] = maxJ

    return res


def get_topography(cross_section, minlo, minla, maxlo, maxla):
    """
    See http://www.trondkristiansen.com/?page_id=846
    """
    toponame = '/Users/mpagani/Data/ETOPO1_Bed_g_gmt4.grd'
    toponame = '/Users/mpagani/Data/globalGTOPO30.grd'
    toponame = '/Users/mpagani/Data/etopo5.nc'

    etopo1 = Dataset(toponame, 'r')

    # ETOPO5
    lons = etopo1.variables["topo_lon"][:]
    lats = etopo1.variables["topo_lat"][:]

    dlt = 0.50
    res = findSubsetIndices(minla-dlt, maxla+dlt,
                            minlo-dlt, maxlo+dlt, lats, lons)

    lon, lat = numpy.meshgrid(lons[int(res[0]):int(res[1])],
                              lats[int(res[2]):int(res[3])])

    print "   Extracted data for area: (%s,%s) to (%s,%s)" % (lon.min(),
                                                              lat.min(),
                                                              lon.max(),
                                                              lat.max())
    bathy = etopo1.variables["topo"][int(res[2]):int(res[3]),
                                     int(res[0]):int(res[1])]

    lonf = lon.flatten()
    latf = lat.flatten()
    batf = bathy.flatten()
    idx = cross_section.get_grd_nodes_within_buffer(lonf, latf, 25.,
                                                    minlo, maxlo, minla, maxla)

    return lonf[idx], latf[idx], batf[idx]


def get_trench_data(filename):
    """
    """
    fin = open(filename, 'r')
    trench = []
    for line in fin:
        aa = re.split('\s+', re.sub('^\s+', '', line))
        trench.append((float(aa[0]), float(aa[1])))
    fin.close()
    trc = numpy.array(trench)


def get_extremes_catalogue(cat):
    """
    """
    midlo = (min(cat.data['longitude'])+max(cat.data['longitude']))/2
    midla = (min(cat.data['latitude'])+max(cat.data['latitude']))/2
    minlo = min(cat.data['longitude'])
    minla = min(cat.data['latitude'])
    maxlo = max(cat.data['longitude'])
    maxla = max(cat.data['latitude'])
    return midlo, midla, minlo, minla, maxlo, maxla


def get_slab1pt0(grdname):
    """
    """
    slab1pt0 = []
    for line in open(grdname):
        if not re.search('^#', line):
            aa = re.split('\s+', line)
            if not re.search('[a-z]', aa[2]):
                slab1pt0.append([float(aa[0]), float(aa[1]), float(aa[2])])
    slab1pt0or = numpy.asarray(slab1pt0)
    return slab1pt0or


def get_slab1pt0sec(csec, slab1pt0or, minlo, maxlo, minla, maxla):
    idx = numpy.nonzero(slab1pt0or[:, 0] > 180)
    slab1pt0 = slab1pt0or
    if len(idx[0]):
        slab1pt0[idx[0], 0] = slab1pt0[idx[0], 0] - 360.
    idxslb = csec.get_grd_nodes_within_buffer(slab1pt0[:, 0],
                                              slab1pt0[:, 1], 2.5,
                                              minlo, maxlo, minla, maxla)
    slab1pt0 = numpy.squeeze(slab1pt0[idxslb, :])
    return slab1pt0


def select_gcmt(tmp_idx_cmt, cat_gcmt):
    idx_cmt = []
    cnt = 0
    for idx in tmp_idx_cmt:
        tmp = cat_gcmt.gcmts[idx].centroid.depth_type
        if re.search(tmp, 'FREE') or re.search(tmp, 'BDY '):
            idx_cmt.append(idx)
            cnt += 1
    return idx_cmt


def gridding(xmin, xmax, ymin, ymax, stepx, stepy, datax, datay, mag):
    """
    """
    idxo = numpy.nonzero((datax > xmin) & (datax < xmax) &
                         (datay > ymin) & (datay < ymax))

    idx = idxo[0].astype(int)
    datax = datax[idx]
    datay = datay[idx]

    resol = 5.0
    x = numpy.arange(numpy.floor(xmin/resol)*resol,
                     numpy.ceil(xmax/resol)*resol+resol, stepx)
    y = numpy.arange(numpy.floor(ymin/resol)*resol,
                     numpy.ceil(ymax/resol)*resol+resol, stepy)
    X, Y = numpy.meshgrid(x, y)
    idx = numpy.round(datax/stepx)
    idy = numpy.round(datay/stepy)

    Zn = numpy.zeros_like(X)
    Zm = numpy.ones_like(X) * 1e-10

    for ix, iy, mag in zip(list(idx), list(idy), mag):
        Zn[int(iy), int(ix)] += 1
        Zm[int(iy), int(ix)] += 10**(1.5*mag+9.05)

    return X, Y, Zn, Zm


def plotting(csidx, csec, olo, ola, strike, lnght, newcat, moho, los, las,
             deps, cat_gcmt, idx_cmt, slab1pt0, other):

    # -----
    # SETTINGS
    max_depth = 250
    ypad = 10
    max_dist = 450
    fig_length = 15

    kaverina = {'N': 'blue',
                'SS': 'green',
                'R': 'red',
                'N-SS': 'turquoise',
                'SS-N': 'palegreen',
                'R-SS': 'goldenrod',
                'SS-R': 'yellow'}

    fig_width = fig_length * (max_depth+ypad) / max_dist

    # -----
    # CREATE THE FIGURE
    fig = plt.figure(figsize=(fig_length, fig_width))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[1, 5])
    # set the spacing between axes
    gs.update(wspace=0.025, hspace=0.05)

    # Axis 1 - Upper right
    ax1 = plt.subplot(gs[1])
    plt.axis('on')
    ax1.set_autoscale_on(False)

    # Axis 2 - Upper right
    ax2 = plt.subplot(gs[2])
    plt.axis('on')
    ax2.set_autoscale_on(False)

    # Axis 3 - Lower Right
    ax3 = plt.subplot(gs[3])
    plt.axis('on')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_autoscale_on(False)

    # Axis 0
    ax0 = plt.subplot(gs[0])
    plt.axis('off')
    ax0.set_autoscale_on(False)

    # -----
    # MAIN PANEL
    ax3 = plt.subplot(gs[3])
    note = 'Cross-Section origin: %.2f %.2f' % (olo, ola)
    plt.gca().annotate(note, xy=(0.0, max_depth+20), xycoords='data',
                       annotation_clip=False, fontsize=8)

    note = 'Cross-Section strike: %.2f' % (strike)
    plt.gca().annotate(note, xy=(0.0, max_depth+30), xycoords='data',
                       annotation_clip=False, fontsize=8)

    note = 'Cross-Section lenght: %.2f [km]' % (lnght)
    plt.gca().annotate(note, xy=(0.0, max_depth+40), xycoords='data',
                       annotation_clip=False, fontsize=8)

    note = 'Cross-Section: %d' % (csidx)
    plt.gca().annotate(note, xy=(500, max_depth+20), xycoords='data',
                       annotation_clip=False, fontsize=8,
                       horizontalalignment='right')
    # -----
    # LEGEND for FOCAL MECHANISMS
    x = 150
    xstep = 30
    y = max_depth+27
    patches = []
    note = 'Rupture mechanism classification (Kaverina et al. 1996)'
    plt.gca().annotate(note, xy=(x, max_depth+20), xycoords='data',
                       annotation_clip=False, fontsize=8)
    for key in sorted(kaverina):
        box = matplotlib.patches.Rectangle(xy=(x, y), width=10, height=10,
                                           color=kaverina[key], clip_on=False)
        plt.gca().annotate(key, xy=(x+12, y+8), annotation_clip=False,
                           fontsize=8)
        x += xstep
        plt.gca().add_patch(box)

    # -----
    # Compute distance between the origin of the cross-section and each
    # selected earthquake
    dsts = geodetic_distance(olo, ola,
                             newcat.data['longitude'],
                             newcat.data['latitude'])
    xg, yg, zgn, zgm = gridding(0, max_dist, 0, max_depth, 5, 5, dsts,
                                newcat.data['depth'][:],
                                newcat.data['magnitude'][:])
    sze = (newcat.data['magnitude'])**0.5
    patches = []
    for dst, dep, mag in zip(dsts, newcat.data['depth'],
                             newcat.data['magnitude']):
        circle = Circle((dst, dep), (mag*0.5)**1.5, ec='white')
        patches.append(circle)
    colors = newcat.data['magnitude']
    p = PatchCollection(patches, zorder=6, edgecolors='white')
    p.set_alpha(0.5)
    p.set_array(numpy.array(colors))
    plt.gca().add_collection(p)
    #   plt.colorbar(p, fraction=0.1)

    # -----
    # Compute histograms
    tmp_mag = newcat.data['magnitude'][:]
    tmp_dep = newcat.data['depth'][:]
    iii = numpy.nonzero((tmp_mag > 3.5) & (tmp_dep > 0.))
    edges_dep = numpy.arange(0, max_depth, 5)
    edges_dist = numpy.arange(0, max_dist, 5)
    seism_depth_hist = scipy.histogram(tmp_dep[iii], edges_dep)
    seism_dist_hist = scipy.histogram(dsts[iii], edges_dist)

    # -----
    # MOHO
    if moho is not None:
        mdsts = geodetic_distance(olo, ola, moho[:, 0], moho[:, 1])
        iii = numpy.argsort(mdsts)
        plt.plot(mdsts[iii], -1*moho[iii, 2], '--p', zorder=10, linewidth=2)

    # -----
    # TOPOGRAPHY
    tdsts = geodetic_distance(olo, ola, los, las)
    iii = numpy.argsort(tdsts)
    plt.plot(tdsts[iii], -1*deps[iii]/1000, '-k', zorder=1, linewidth=2)

    # -----
    # Focal mechanisms
    cmt_dst = geodetic_distance(olo, ola, cat_gcmt.data['longitude'][idx_cmt],
                                cat_gcmt.data['latitude'][idx_cmt])
    cmt_dep = cat_gcmt.data['depth'][idx_cmt]
    cmts = numpy.array(cat_gcmt.gcmts)[idx_cmt]
    ax = plt.gca()

    for idx, ddd, dep, eve, mag, yea in zip(idx_cmt, list(cmt_dst),
                list(cmt_dep), list(cmts), cat_gcmt.data['magnitude'][idx_cmt],
                cat_gcmt.data['year'][idx_cmt]):

        if yea > 1000 and mag > 1.0:

            # This gets the ka
            plungeb = cat_gcmt.data['plunge_b'][idx]
            plungep = cat_gcmt.data['plunge_p'][idx]
            plunget = cat_gcmt.data['plunge_t'][idx]
            mclass = mecclass(plunget, plungeb, plungep)

            com = eve.moment_tensor._to_6component()
            eve.nodal_planes.nodal_plane_1
            bcc = beach(com, xy=(ddd, dep), width=eve.magnitude*3,
                        linewidth=1, zorder=20, size=mag,
                        facecolor=kaverina[mclass])
            bcc.set_alpha(0.5)
            ax.add_collection(bcc)

            plot_planes_at(ddd, dep,
                           [eve.nodal_planes.nodal_plane_1['strike'],
                            eve.nodal_planes.nodal_plane_2['strike']],
                           [eve.nodal_planes.nodal_plane_1['dip'],
                            eve.nodal_planes.nodal_plane_2['dip']],
                           [mag, mag], strike, 90.,
                           aratio=1.0, color=kaverina[mclass], linewidth=2.0)

    gcmt_dist_hist = scipy.histogram(cmt_dst, edges_dist)

    # -----
    # SLAB 1.0
    slb_dst = geodetic_distance(olo, ola, slab1pt0[:, 0], slab1pt0[:, 1])
    slb_dep = slab1pt0[:, 2]
    iii = numpy.argsort(slb_dst)
    plt.plot(slb_dst[iii], -1*slb_dep[iii], '-g', linewidth=3, zorder=30)
    if len(slb_dst):
        plt.text(slb_dst[iii[-1]], -1*slb_dep[iii[-1]], 'Slab1.0', fontsize=8)

    # -----
    # OTHER
    if 'kyriakor' in other:
        kyriakor = other['kyriakor']
        kyr_dst = geodetic_distance(olo, ola, kyriakor[:, 0], kyriakor[:, 1])
        kyr_dep = kyriakor[:, 2]
        iii = numpy.argsort(kyr_dst)
        plt.plot(kyr_dst[iii], kyr_dep[iii], '--g', linewidth=2, zorder=30)

    # -----
    # Adding the colorbar
    # See http://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    axes = []
    for gg in gs:
        axes.append(plt.subplot(gg))
    fig.colorbar(p, ax=axes, fraction=0.01, pad=0.01)

    # -----
    # Final adjustments
    ax3.invert_yaxis()
    plt.xlim([0, 500])
    ax3.grid(which='both', zorder=20)
    # Axis limits
    plt.ylim([max_depth, -ypad])
    xli = ax3.get_xlim()
    yli = ax3.get_ylim()

    # -----
    # DEPTH PANEL
    ax2 = plt.subplot(gs[2])
    plt.barh(edges_dep[:-1], seism_depth_hist[0],
             height=numpy.diff(edges_dep)[0], facecolor='none')
    plt.ylabel('Depth [km]')
    ax2.grid(which='both', zorder=20)
    xmax = numpy.ceil(max(seism_depth_hist[0])/10.)*10.
    ax2.set_xlim([0, xmax+xmax*0.05])
    ax2.invert_xaxis()
    ax2.set_ylim(yli)

    # -----
    # DISTANCES PANEL
    ax1 = plt.subplot(gs[1])
    plt.bar(edges_dist[:-1], seism_dist_hist[0],
            width=numpy.diff(edges_dist)[0], facecolor='none')
    plt.bar(edges_dist[:-1], gcmt_dist_hist[0],
            width=numpy.diff(edges_dist)[0], facecolor='red', alpha=0.4)
    plt.xlim(xli)
    # Moving ticks on top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    ax1.set_axisbelow(False)
    plt.xlabel('Distance from the trench [km]')
    ax1.grid(which='both', zorder=20)
    ymax = numpy.ceil(max(seism_dist_hist[0])/10.)*10.
    ax1.set_ylim([0, ymax+ymax*0.05])

    # -----
    # SAVING FIGURE
    outname = './figs/%d_csec.png' % (csidx)
    plt.savefig(outname)
    plt.close()


def main():
    """
    """
    # Load trench data
    filename_trench = './trench.xy'
    #  trc = get_trench_data(filename_trench)
    # Load catalogue
    cat = pickle.load(open("catalogue_ext_cac.p", "rb"))
    # Catalogue selector
    selector = CatalogueSelector(cat, create_copy=True)
    # SLAB 1.0
    grdname = "/Users/mpagani/Data/subduction/mex_slab1.0_clip.xyz"
    slab1pt0or = get_slab1pt0(grdname)
    # Kyriakopoulos 2015
    grdname = "/Users/mpagani/Data/subduction/Kyriakopoulos_etal_JGR_2015_slabInterface.xyz"
    kyriakor  = get_slab1pt0(grdname)
    # GCMT
    gcmt_filename = '/Users/mpagani/Data/catalogues/gcmt/jan76_dec13.ndk'
    parser = ParseNDKtoGCMT(gcmt_filename)
    cat_gcmt = parser.read_file()
    # Load cross-sections
    fin = open('cs_traces.csv')
    # Process traces
    for idx, line in enumerate(fin):

        # if idx > 16: break

        if idx < 1:
            continue
        print 'CROSS-SECTION: ', idx

        # Cross-Section parameters
        aa = line.split(',')
        olo = float(aa[1])
        ola = float(aa[2])
        lnght = 500.
        strike = float(aa[3])
        strike = 25

        # Create cross-section
        csec = CrossSection(olo, ola, [lnght], [strike])

        # Selected earthquakes
        idxs = csec.get_eqks_within_buffer(cat, 50.)
        print '   Number of selected eqks:', len(idxs[0])

        # Create subcatalogue
        boo = numpy.zeros_like(cat.data['magnitude'], dtype=int)
        boo[idxs] = 1
        newcat = selector.select_catalogue(boo)

        dlt = 0.0
        minlo = min(csec.plo)-dlt
        maxlo = max(csec.plo)+dlt
        minla = min(csec.pla)-dlt
        maxla = max(csec.pla)+dlt

        # Getting MOHO
        moho = get_crust1pt0_moho_depth(csec, minlo, maxlo, minla, maxla)
        if moho is not None:
            moho = numpy.squeeze(moho)

        # Topography
        los, las, deps = get_topography(csec, minlo, minla, maxlo, maxla)

        # SLAB 1.0
        tmp_idx_cmt = get_cmt(cat_gcmt, csec, minlo, maxlo, minla, maxla)

        idx_cmt = select_gcmt(tmp_idx_cmt, cat_gcmt)
        slab1pt0 = get_slab1pt0sec(csec, slab1pt0or, minlo, maxlo, minla,
                                   maxla)

        # Other
        kyriakor0 = get_slab1pt0sec(csec, kyriakor, minlo, maxlo, minla, maxla)

        # Plotting
        plotting(idx, csec, olo, ola, strike, lnght, newcat, moho,  los, las, deps,
                 cat_gcmt, idx_cmt, slab1pt0, {'kyriakor': kyriakor0})

if __name__ == "__main__":
    main()
