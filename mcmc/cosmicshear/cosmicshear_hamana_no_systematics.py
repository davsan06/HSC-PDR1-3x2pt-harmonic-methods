#!/usr/bin/env python

import os
import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
import sacc

# Sources
"""
    Creating sources, each one maps to a specific section of a SACC file. In
    this case trc0, trc1 describe weak-lensing probes. The sources are saved
    in a dictionary since they will be used by one or more two-point function.
"""
sources = {}
# nlasystematic = wl.NonLinearAlignmentSystematic(sacc_tracer="")

for i in range(1,5):
    """
    We include a photo-z shift bias (a constant shift in dndz). We also
    have a different parameter for each bin, so here again we use the
    src{i}_ prefix.
    """
    # multbias = wl.HSCMultiplicativeShearBias(sacc_tracer="")
    # pzshift = wl.SourcePhotoZShift(sacc_tracer=f"source_{i}")
    # pzstretch = wl.SourcePhotoZStretch(sacc_tracer=f"source_{i}")

    """
        Now we can finally create the weak-lensing source that will compute the
        theoretical prediction for that section of the data, given the
        systematics.
    """
    # sources[f"source_{i}"] = wl.WeakLensing(sacc_tracer=f"source_{i}", systematics=[multbias,pzshift,pzstretch,nlasystematic])
    # sources[f"source_{i}"] = wl.WeakLensing(sacc_tracer=f"source_{i}", systematics=[multbias,pzshift,nlasystematic])
    sources[f"source_{i}"] = wl.WeakLensing(sacc_tracer=f"source_{i}")

"""
    Now that we have all sources we can instantiate all the two-point
    functions. For each one we create a new two-point function object.
"""
stats = {}

"""
    Creating all auto/cross-correlations two-point function objects for
    the weak-lensing probes.
"""
# Example https://github.com/LSSTDESC/firecrown/blob/master/examples/des_y1_3x2pt/des_y1_3x2pt.py
for i in range(1,5):
    for j in range(1,5):
        if i >= j:
            # Xi plus
            stats[f"xip_src{j}_src{i}"] = TwoPoint(
                source0=sources[f"source_{j}"],
                source1=sources[f"source_{i}"],
                sacc_data_type="galaxy_shear_xi_plus",
            )
            # Xi minus
            stats[f"xim_src{j}_src{i}"] = TwoPoint(
                source0=sources[f"source_{j}"],
                source1=sources[f"source_{i}"],
                sacc_data_type="galaxy_shear_xi_minus",
            )

"""
    Here we instantiate the actual likelihood. The statistics argument carry
    the order of the data/theory vector.
"""
lk = ConstGaussian(statistics=list(stats.values()))

"""
    We load the correct SACC file.
"""
# saccfile = os.path.expanduser(
#     os.path.expandvars("/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_real_Hamana_shear.sacc")
# )
saccfile = os.path.expanduser(os.path.expandvars('/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_real_raw_hsc_HamanaShearSC.sacc'))

sacc_data = sacc.Sacc.load_fits(saccfile)

"""
    The read likelihood method is called passing the loaded SACC file, the
    two-point functions will receive the appropriated sections of the SACC
    file and the sources their respective dndz.
"""
lk.read(sacc_data)

# Genero cosmologia ccl
# cosmo = pyccl.Cosmology(Omega_c=0.2916,
#                         Omega_b=0.03636,
#                         h=0.786003,
#                         n_s=0.989329,
#                         sigma8=0.772384,
#                         transfer_function='bbks')
# lk.compute_chisq(cosmo)

"""
   This script will be loaded by the appropriated connector. The framework
    then looks for the `likelihood` variable to find the instance that will
    be used to compute the likelihood.
"""
likelihood = lk
