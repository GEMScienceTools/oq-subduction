plot_cross_section.py can now plot the original cross section, or the classified hypocenters and the picked cross section.
To use for classification, add to ini file:

[general]

type = classification

#all other types will be ignored

[data]

class_base = <path to classified earthquake csv files>

class_list = <list of classified earthquake files separated by commas; names contain 'crustal', 'int', 'slab', and 'unc'>

#if exists
cross_section_directory = <path to picked cross section with name cs_%d.csv % id>

#the cross section path can be specified for the original cross section plotting, as well
