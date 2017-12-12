This branch contains the following changes:
(1) changed resampling algorithm in create_multiple_cross_sections.py. cross section origin points now fall on the trench axis and have azimuth of original trench (instead of resampled trench). Includes cross sections for full trench (former version missed the last cross section along a trench)
(2) updates in cross_section.py, create_multiple_cross_sections.py, plot_cross_section.py, plot_2pt5_model.py to make sure that cross sections straddling the international date line are plotted with the right selection of data (e.g., the parser collects data for the shortest path between the cross section end points) 
(3) properly outputs picked depths for interactive part of plot_cross_section.py, in which the user defines the interface. Former version did not clear data when user started over.
(4) edits to plot_cross_section.py (through cross_section.py) to force plot to continue when lithosphere, moho, topography, slab, etc. have no data or only one point within the cross section swath
(5) new projection function used in create_2pt5_model.py for interpolating along interface profiles - works everywhere
