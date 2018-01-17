#!/usr/bin/env python

import os
import sys

from oq.hmtk.subduction.edges_set import EdgesSet

from openquake.hazardlib.sourcewriter import write_source_model


def complex_fault_src_from_edges(edges_folder, out_nrml='source.xml'):
    """
    :param edges_folder:
    """
    #
    # check edges folder
    assert os.path.exists(edges_folder)
    #
    #
    es = EdgesSet.from_files(edges_folder)
    src = es.get_complex_fault()
    print(out_nrml)
    write_source_model(out_nrml, [src], 'Name')


def runner(argv):

    opt = 1
    if opt == 0:
        fname = '/Users/mpagani/NC/Hazard_Charles/Hazard_models/cc18/data/sources_subduction/profiles/sp_lan_int'
        outfile = '/Users/mpagani/NC/Hazard_Charles/Hazard_models/ccar18/tmp_sub/nrml/int_lan.xml'

    if opt == 1:
        fname = '/Users/mpagani/NC/Hazard_Charles/Hazard_models/cc18/data/sources_subduction/profiles/sp_cam_int'
        outfile = '/Users/mpagani/NC/Hazard_Charles/Hazard_models/ccar18/tmp_sub/nrml/int_cam.xml'

    complex_fault_src_from_edges(fname, outfile)


if __name__ == "__main__":
    runner(sys.argv[1:])
