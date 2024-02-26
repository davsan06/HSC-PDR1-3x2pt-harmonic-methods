# Use desc-python-bleed to avoid Latex issues
from ast import If
import os
import sys

from numpy.lib.npyio import save
sys.path.insert(0, '/global/homes/d/davidsan/ChainConsumer')
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('text', usetex=True)
# plt.style.use('/global/cscratch1/sd/davidsan/dsc_custom.mplstyle')
import scipy.stats as stats
from numpy.random import normal, uniform
from scipy.special import ndtri

# Define matplolib colors: black, purple, green, red
colors = ['#000000', '#800080', '#008000', '#ff0000', "#E69F00", "#56B4E9", "#009E73", "#F0E442", '#800080', "#0072B2", "#CC79A7", "#D55E00"]

# Matplotlib style
plt.rcParams['figure.figsize'] = 8., 6.
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.subplot.left'] = 0.125
plt.rcParams['figure.subplot.right'] = 0.9
plt.rcParams['figure.subplot.bottom'] = 0.125
plt.rcParams['figure.subplot.top'] = 0.9
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.pad'] = 6.
plt.rcParams['xtick.minor.pad'] = 6.
plt.rcParams['ytick.major.pad'] = 6.
plt.rcParams['ytick.minor.pad'] = 6.
plt.rcParams['xtick.major.size'] = 4. # major tick size in points
plt.rcParams['xtick.minor.size'] = 3. # minor tick size in points
plt.rcParams['ytick.major.size'] = 4. # major tick size in points
plt.rcParams['ytick.minor.size'] = 3. # minor tick size in points
# Thickness of the axes lines
plt.rcParams['axes.linewidth'] = 1.5
# Smaller font size for axes ticks labels
plt.rcParams['xtick.labelsize'] = 13
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] =  'serif'
# plt.rcParams['font.serif'] = 'Computer Modern Roman Bold'
plt.rcParams['font.size'] = 18  

# Chainconsumer smooth parameter
# kde = 3.0
kde = 1.3

##############################################
### Dictionaries and parameters definition ###
##############################################
parameters_desc = [# Cosmology parameters
                   '$S_8$',
                   '$\Omega_m$',
                   '$\Omega_{cdm} \cdot h^2$',
                   '$\Omega_b \cdot h^2$',
                   '$\\Omega_\\nu h^2$',
                   '$h$',
                   '$A_s$',
                   '$\ln(10^{10} A_s)$',
                   '$n_{s}$',
                   # Lens photo-z uncert.
                   '$\Delta z^{lens}_1$',
                   '$\Delta z^{lens}_2$',
                   '$\Delta z^{lens}_3$',
                   '$\Delta z^{lens}_4$',
                   '$\sigma z^{lens}_1$',
                   '$\sigma z^{lens}_2$',
                   '$\sigma z^{lens}_3$',
                   '$\sigma z^{lens}_4$',
                   # Lens galaxy bias 
                   '$b^{lens}_1$',
                   '$b^{lens}_2$',
                   '$b^{lens}_3$',
                   '$b^{lens}_4$',
                   '$\sigma_8$',
                   '$\sigma_12$',
                   # Sources photo-z uncert.
                   '$\Delta z^{source}_1$',
                   '$\Delta z^{source}_2$',
                   '$\Delta z^{source}_3$',
                   '$\Delta z^{source}_4$',
                   '$\sigma z^{source}_1$',
                   '$\sigma z^{source}_2$',
                   '$\sigma z^{source}_3$',
                   '$\sigma z^{source}_4$',
                   # Multiplicative shear bias
                   'm',
                   # Non-linear intrinsic alignment
                   '$A_{IA}$',
                   '$\eta$',
                   # Linear Intrinsic Alignment
                   '$A_{IA,lin}$',
                   '$\\alpha_z$',
                   # Stats
                   '$\Chi^2$',
                   '$prior$',
                   '$like$',
                   '$post$',
                   '$weight$']

# Dict = {cosmosis_name:latex_label}
parameters_dict_desc = {# Cosmology parameters
                        'cosmological_parameters--omch2':'$\Omega_{cdm} \cdot h^2$',
                        'cosmological_parameters--omega_c':'$\Omega_{cdm}$',
                        'cosmological_parameters--ombh2':'$\Omega_b \cdot h^2$',
                        'cosmological_parameters--omega_b':'$\Omega_b$',
                        'cosmological_parameters--omnuh2':'$\\Omega_\\nu h^2$',
                        'cosmological_parameters--h0':'$h$',
                        'cosmological_parameters--a_s':'$A_s$',
                        'cosmological_parameters--log10as':'$\ln(10^{10} A_s)$',
                        'cosmological_parameters--log10as_hamana':'$\ln(10^{9} A_s)$',
                        'cosmological_parameters--n_s':'$n_{s}$',
                        'cosmological_parameters--w':'w',
                        'cosmological_parameters--sigma_8':'$\sigma_8$',
                        'COSMOLOGICAL_PARAMETERS--SIGMA_8':'$\sigma_8$',
                        'COSMOLOGICAL_PARAMETERS--SIGMA_12':'$\sigma_12$',
                        'COSMOLOGICAL_PARAMETERS--S_8':'$S_8$',
                        'cosmological_parameters--omega_m':'$\Omega_m$',
                        'COSMOLOGICAL_PARAMETERS--OMEGA_M':'$\Omega_m$',
                        # Halo model:
                        'halo_model_parameters--a_bary':'$a_{bary}$',
                        # Lens photo-z uncert.
                        'firecrown_two_point--lens_0_delta_z':'$\Delta z^{lens}_1$',
                        'firecrown_two_point--lens_1_delta_z':'$\Delta z^{lens}_2$',
                        'firecrown_two_point--lens_2_delta_z':'$\Delta z^{lens}_3$',
                        'firecrown_two_point--lens_3_delta_z':'$\Delta z^{lens}_4$',
                        'firecrown_two_point--lens_0_sigma_z':'$\sigma z^{lens}_1$',
                        'firecrown_two_point--lens_1_sigma_z':'$\sigma z^{lens}_2$',
                        'firecrown_two_point--lens_2_sigma_z':'$\sigma z^{lens}_3$',
                        'firecrown_two_point--lens_3_sigma_z':'$\sigma z^{lens}_4$',
                        # Lens galaxy bias 
                        'firecrown_two_point--lens_0_bias':'$b^{lens}_1$',
                        'firecrown_two_point--lens_1_bias':'$b^{lens}_2$',
                        'firecrown_two_point--lens_2_bias':'$b^{lens}_3$',
                        'firecrown_two_point--lens_3_bias':'$b^{lens}_4$',
                        # Photo-z WL
                        'firecrown_two_point--source_0_delta_z':'$\Delta z^{source}_1$',
                        'firecrown_two_point--source_1_delta_z':'$\Delta z^{source}_2$',
                        'firecrown_two_point--source_2_delta_z':'$\Delta z^{source}_3$',
                        'firecrown_two_point--source_3_delta_z':'$\Delta z^{source}_4$',
                        'firecrown_two_point--source_4_delta_z':'$\Delta z^{source}_4$',
                        'firecrown_two_point--source_0_sigma_z':'$\sigma z^{source}_1$',
                        'firecrown_two_point--source_1_sigma_z':'$\sigma z^{source}_2$',
                        'firecrown_two_point--source_2_sigma_z':'$\sigma z^{source}_3$',
                        'firecrown_two_point--source_3_sigma_z':'$\sigma z^{source}_4$',
                        # Photo-z WL HSC Year 3
                        'nz_sample_errors--bias_1':'$\Delta z^{source}_1$',
                        'nz_sample_errors--bias_2':'$\Delta z^{source}_2$',
                        'nz_sample_errors--bias_3':'$\Delta z^{source}_3$',
                        'nz_sample_errors--bias_4':'$\Delta z^{source}_4$',
                        'wl_photoz_errors--bias_1':'$\Delta z^{source}_1$',
                        'wl_photoz_errors--bias_2':'$\Delta z^{source}_2$',
                        'wl_photoz_errors--bias_3':'$\Delta z^{source}_3$',
                        'wl_photoz_errors--bias_4':'$\Delta z^{source}_4$',
                        # PSF Systematics HSC Year 3
                        'psf_systematics_parameters--psf_cor1_z1':'$a_{psf}$',
                        'psf_systematics_parameters--psf_cor2_z1':'$b_{psf}$',
                        'psf_systematics_parameters--psf_cor3_z1':'$c_{psf}$',
                        'psf_systematics_parameters--psf_cor4_z1':'$d_{psf}$',
                        'psf_parameters--psf_alpha2':'$\\alpha_2$',
                        'psf_parameters--psf_beta2':'$\\beta_2$',
                        'psf_parameters--psf_alpha4':'$\\alpha_4$',
                        'psf_parameters--psf_beta4':'$\\beta_4$',
                        # Multiplicative shear bias.
                        'firecrown_two_point--mult_bias':'m',
                        # Shear calibration parameters HSC Year 3
                        'shear_calibration_parameters--m1':'m1',
                        'shear_calibration_parameters--m2':'m2',
                        'shear_calibration_parameters--m3':'m3',
                        'shear_calibration_parameters--m4':'m4',
                        # Non-linear Intrinsic Alignment
                        'firecrown_two_point--a_ia':'$A_{IA}$',
                        'firecrown_two_point--eta_eff':'$\eta$',
                        'intrinsic_alignment_parameters--a1':'$a_1$',
                        'intrinsic_alignment_parameters--a2':'$a_2$',
                        'intrinsic_alignment_parameters--alpha1':'$\\alpha_1$',
                        'intrinsic_alignment_parameters--alpha2':'$\\alpha_2$',
                        'intrinsic_alignment_parameters--bias_ta':'$bias_{ta}$',
                        # Linear Intrinsic Alignment
                        'firecrown_two_point--ia_bias':'$A_{IA,lin}$',
                        'firecrown_two_point--alphaz':'$\\alpha_z$',
                        # Stats
                        'DATA_VECTOR--2PT_CHI2':'$\Chi^2$',
                        'prior':'$prior$',
                        'like':'$like$',
                        'post':'$post$',
                        'weight\n':'$weight$'}

parameters_dict_desc_hamana = {# Cosmology parameters
                                'cosmological_parameters--omch2':'$\Omega_{cdm} \cdot h^2$',
                                'cosmological_parameters--omega_c':'$\Omega_{cdm}$',
                                'cosmological_parameters--ombh2':'$\Omega_b \cdot h^2$',
                                'cosmological_parameters--omega_b':'$\Omega_b$',
                                'cosmological_parameters--h0':'$h$',
                                'cosmological_parameters--a_s':'$A_s$',
                                'cosmological_parameters--log10as':'$\ln(10^{10} A_s)$',
                                'cosmological_parameters--log10as_hamana':'$\ln(10^{9} A_s)$',
                                'cosmological_parameters--n_s':'$n_{s}$',
                                'COSMOLOGICAL_PARAMETERS--SIGMA_8':'$\sigma_8$',
                                'COSMOLOGICAL_PARAMETERS--SIGMA_12':'$\sigma_12$',
                                'cosmological_parameters--sigma_8':'$\sigma_8$',
                                # Photo-z WL
                                'firecrown_two_point--source_1_delta_z':'$\Delta z^{source}_1$',
                                'firecrown_two_point--source_2_delta_z':'$\Delta z^{source}_2$',
                                'firecrown_two_point--source_3_delta_z':'$\Delta z^{source}_3$',
                                'firecrown_two_point--source_4_delta_z':'$\Delta z^{source}_4$',
                                # Multiplicative shear bias.
                                'firecrown_two_point--mult_bias':'m',
                                # Non-linear Intrinsic Alignment
                                'firecrown_two_point--a_ia':'$A_{IA}$',
                                'firecrown_two_point--eta_eff':'$\eta$',
                                # Linear Intrinsic Alignment
                                'firecrown_two_point--ia_bias':'$A_{IA,lin}$',
                                'firecrown_two_point--alphaz':'$\\alpha_z$',
                                # Stats
                                'DATA_VECTOR--2PT_CHI2':'$\Chi^2$',
                                'prior':'$prior$',
                                'like':'$like$',
                                'post':'$post$',
                                'weight\n':'$weight$'}

extents_dict = {
                # Cosmology parameters
               '$S_8$':[0.5,1.0],
               '$\Omega_m$':[0.0,0.7],
               '$\Omega_{cdm} \cdot h^2$':[0.03,0.7],
               '$\Omega_b \cdot h^2$':[0.019,0.026],
               '$h$':[0.6,0.9],
               '$A_s$':[],
               '$\ln(10^{10} A_s)$':[1.5,6.0],
               '$n_{s}$':[0.87,1.07],
               '$\sigma_8$':[0.45,1.6],
               'w':[-3.0, -0.333],
               # Lens photo-z uncert.
               '$\Delta z^{lens}_1$':[-1.05, 1.05],
               '$\Delta z^{lens}_2$':[-1.05, 1.05],
               '$\Delta z^{lens}_3$':[-1.05, 1.05],
               '$\Delta z^{lens}_4$':[-1.05, 1.05],
               '$\sigma z^{lens}_1$':[0.75,1.25],
               '$\sigma z^{lens}_2$':[0.75,1.25],
               '$\sigma z^{lens}_3$':[0.75,1.25],
               '$\sigma z^{lens}_4$':[0.75,1.25],
               # Lens galaxy bias 
               '$b^{lens}_1$':[0.1,6.0],
               '$b^{lens}_2$':[0.1,6.0],
               '$b^{lens}_3$':[0.1,6.0],
               '$b^{lens}_4$':[0.1,6.0],
               # Sources photo-z uncert.
               '$\Delta z^{source}_1$':[-0.1,0.1],
               '$\Delta z^{source}_2$':[-0.1,0.1],
               '$\Delta z^{source}_3$':[-0.1,0.1],
               '$\Delta z^{source}_4$':[-0.1,0.1],
               # Multiplicative shear bias
               'm':[-0.025,0.025],
               # Non-linear intrinsic alignment
               '$A_{IA}$':[-5.0,5.0],
               '$\eta$':[-5.0,5.0]

}
parameters_hsc = ['$weight$',
                  '$\log{\mathcal{L}}$',
                  '$\Omega_b \cdot h^2$',
                  '$\Omega_{cdm} \cdot h^2$',
                  '$n_{s}$',
                  '$h$',
                  '$\ln(10^{10} A_s)$',
                  '$A_{IA}$',
                  '$\eta$',
                  'm',
                  '$a_{psf}$',
                  '$b_{psf}$',
                  '$\Delta z^{source}_1$',
                  '$\Delta z^{source}_2$',
                  '$\Delta z^{source}_3$',
                  '$\Delta z^{source}_4$',
                  '$\Omega_m$',
                  '$\sigma_8$']
parameters_hsc_hamana = ['$weight$',
                         '$\log{\mathcal{L}}$',
                         '$\Omega_{cdm}$',
                         '$\ln(10^{9} A_s)$',
                         '$\Omega_b$',
                         '$n_{s}$',
                         '$h$',
                         '$A_{IA}$',
                         '$\eta$',
                         '$\Delta z^{source}_1$',
                         '$\Delta z^{source}_2$',
                         '$\Delta z^{source}_3$',
                         '$\Delta z^{source}_4$',
                         '$a_{psf}$',
                         '$b_{psf}$',
                         'm',
                         '$\Omega_m$',
                         '$\sigma_8$',
                         '$S_8$',
                         '$S_8 (\alpha = 0.45)$',
                         '$\Omega_L$',
                         '$\theta_*$',
                         '$\Omega_\nu$']

parameters_nicola = ['$weight$',
                     '$\log{\mathcal{L}}$',
                     '$A_s \times 10^9$',
                     '$\Omega_{cdm}$',
                     '$\Omega_b$',
                     '$h$',
                     '$n_{s}$',
                     'm1',
                     'm2',
                     'm3',
                     'm4',
                     '$\Delta z^{source}_1$',
                     '$\Delta z^{source}_2$',
                     '$\Delta z^{source}_3$',
                     '$\Delta z^{source}_4$',
                     '$A_{IA}$',
                     '$\eta$',
                     '$\sigma_8$',
                     '$\log{\mathcal{P}}$',
                     '$\log{\mathcal{P}} bis$',
                     '$\Chi^2$',
                     '$\Chi^2$ bis',]

parameters_mn_eqwpost = ['$\Omega_{cdm} \cdot h^2$',
                         '$\Omega_b \cdot h^2$',
                         '$h$', 
                         '$\ln(10^{10} A_s)$',
                         '$n_{s}$',
                         '$\Delta z^{source}_1$',
                         '$\Delta z^{source}_2$',
                         '$\Delta z^{source}_3$',
                         '$\Delta z^{source}_4$',
                         'm',
                         '$A_{IA}$',
                         '$\eta$',
                         '$\sigma_8$',
                         '$\Chi^2$',
                         '$prior$',
                         '$like$',
                         '$post$'
                          ]

parameters_mn_eqwpost_new = ['$\Omega_{cdm} \cdot h^2$',
                             '$\Omega_b \cdot h^2$',
                             '$h$', 
                             '$\ln(10^{10} A_s)$',
                             '$n_{s}$',
                             '$\Delta z^{source}_1$',
                             '$\Delta z^{source}_2$',
                             '$\Delta z^{source}_3$',
                             '$\Delta z^{source}_4$',
                             'm',
                             '$A_{IA}$',
                             '$\eta$',
                             '$\Omega_m$',
                             '$\sigma_8$',
                             '$S_8$',
                             '$\Chi^2$',
                             '$prior$',
                             '$like$',
                             '$post$'
                              ]

parameters_mn_eqwpost_no_syst = ['$\Omega_{cdm} \cdot h^2$',
                                 '$\Omega_b \cdot h^2$',
                                 '$h$', 
                                 '$\ln(10^{10} A_s)$',
                                 '$n_{s}$',
                                 '$\sigma_8$',
                                 '$\Chi^2$',
                                 '$prior$',
                                 '$like$',
                                 '$post$'
                                  ]

parameters_mn_eqwpost_hamana_no_syst = [ '$\Omega_{cdm}$',
                                         '$\Omega_b$',
                                         '$\ln(10^{9} A_s)$',
                                         '$n_{s}$',
                                         '$h$',
                                         '$\sigma_8$',
                                         '$\Chi^2$',
                                         '$prior$',
                                         '$like$',
                                         '$post$' ]

parameters_mn_eqwpost_hamana = [ '$\Omega_{cdm}$',
                                 '$\Omega_b$',
                                 '$\ln(10^{9} A_s)$',
                                 '$n_{s}$',
                                 '$h$',
                                 '$\Delta z^{source}_1$',
                                 '$\Delta z^{source}_2$',
                                 '$\Delta z^{source}_3$',
                                 '$\Delta z^{source}_4$',
                                 'm',
                                 '$A_{IA}$',
                                 '$\eta$',
                                 '$\sigma_8$',
                                 '$\Chi^2$',
                                 '$prior$',
                                 '$like$',
                                 '$post$' ]

parameters_des_3x2pt = ["$\Omega_m$",
                            "$h$",
                            '$\Omega_b$',
                            "$n_s$",
                            "$A_s$",
                            "$\Omega_\nu h^2$",
                            "$m_1$",
                            "$m_2$",
                            "$m_3$",
                            "$m_4$",
                            "$zs_bias\_1$",
                            "$zs_bias\_2$",
                            "$zs_bias\_3$",
                            "$zs_bias\_4$",
                            "$zl_bias\_1$",
                            "$zl_bias\_2$",
                            "$zl_bias\_3$",
                            "$zl_bias\_4$",
                            "$zl_bias\_5$",
                            "$zl_width\_1$",
                            "$zl_width\_2$",
                            "$zl_width\_3$",
                            "$zl_width\_4$",
                            "$zl_width\_5$",
                            "$zl_bias\_6$",
                            "$zl_width\_6$",
                            "$b1e\_sig8\_bin1$",
                            "$b1e\_sig8\_bin2$",
                            "$b1e\_sig8\_bin3$",
                            "$b1e\_sig8\_bin4$",
                            "$b1e\_sig8\_bin5$",
                            "$b1e\_sig8\_bin6$",
                            "$b2e\_sig8sq\_bin1$",
                            "$b2e\_sig8sq\_bin2$",
                            "$b2e\_sig8sq\_bin3$",
                            "$b2e\_sig8sq\_bin4$",
                            "$b2e\_sig8sq\_bin5$",
                            "$b2e\_sig8sq\_bin6$",
                            "$a1$",
                            "$a2$",
                            "$alpha1$",
                            "$alpha2$",
                            "$bias\_ta$",
                            '$\\sigma_8$',
                            "$SIGMA\_12$",
                            "$2PT\_CHI2$",
                            "$prior$",
                            "$like$",
                            "$post$",
                            "$weight$"
        ]

parameters_des_2x2pt = [
                       "$\Omega_m$",
                        r"$h_0$",
                        r"$\omega_b$",
                        r"$n_s$",
                        r"$a_s$",
                        r"$\Omega_{\nu} h^2$",
                        r"$m_1$",
                        r"$m_2$",
                        r"$m_3$",
                        r"$m_4$",
                        r"$\text{bias\_1}^{\text{wl}}$",
                        r"$\text{bias\_2}^{\text{wl}}$",
                        r"$\text{bias\_3}^{\text{wl}}$",
                        r"$\text{bias\_4}^{\text{wl}}$",
                        r"$\text{bias\_1}^{\text{lens}}$",
                        r"$\text{bias\_2}^{\text{lens}}$",
                        r"$\text{bias\_3}^{\text{lens}}$",
                        r"$\text{bias\_4}^{\text{lens}}$",
                        r"$\text{width\_1}^{\text{lens}}$",
                        r"$\text{width\_2}^{\text{lens}}$",
                        r"$\text{width\_3}^{\text{lens}}$",
                        r"$\text{width\_4}^{\text{lens}}$",
                        r"$b1e\_sig8\_bin1$",
                        r"$b1e\_sig8\_bin2$",
                        r"$b1e\_sig8\_bin3$",
                        r"$b1e\_sig8\_bin4$",
                        r"$b2e\_sig8sq\_bin1$",
                        r"$b2e\_sig8sq\_bin2$",
                        r"$b2e\_sig8sq\_bin3$",
                        r"$b2e\_sig8sq\_bin4$",
                        r"$a1$",
                        r"$a2$",
                        r"$\alpha_1$",
                        r"$\alpha_2$",
                        r"$\text{bias\_ta}$",
                        '$\\sigma_8$',
                        r"$\Sigma_{12}$",
                        r"$\text{2PT\_CHI2}$",
                        r"$\text{prior}$",
                        r"$\text{like}$",
                        r"$\text{post}$",
                        "$weight$"
]


parameters_des_1x2pt = [
                        "$\Omega_m$",
                        r"$h$",
                        r"$\omega_b$",
                        r"$n_s$",
                        r"$a_s$",
                        r"$\Omega_{\nu}h^2$",
                        r"$m_1$",
                        r"$m_2$",
                        r"$m_3$",
                        r"$m_4$",
                        r"$\text{bias\_1}^{\text{wl}}$",
                        r"$\text{bias\_2}^{\text{wl}}$",
                        r"$\text{bias\_3}^{\text{wl}}$",
                        r"$\text{bias\_4}^{\text{wl}}$",
                        r"$\text{bias\_1}^{\text{lens}}$",
                        r"$\text{bias\_2}^{\text{lens}}$",
                        r"$\text{bias\_3}^{\text{lens}}$",
                        r"$\text{width\_1}^{\text{lens}}$",
                        r"$\text{width\_2}^{\text{lens}}$",
                        r"$\text{width\_3}^{\text{lens}}$",
                        r"$b_1^{\text{lens}}$",
                        r"$b_2^{\text{lens}}$",
                        r"$b_3^{\text{lens}}$",
                        r"$a_1^{\text{ia}}$",
                        r"$a_2^{\text{ia}}$",
                        r"$\alpha_1^{\text{ia}}$",
                        r"$\alpha_2^{\text{ia}}$",
                        r"$\text{bias\_ta}^{\text{ia}}$",
                        '$\\sigma_8$',
                        r"$\Sigma_{12}$",
                        r"$\text{2PT\_CHI2}$",
                        r"$\text{prior}$",
                        r"$\text{like}$",
                        r"$\text{post}$",
                        "$weight$"
]

parameters_des_kids = ['$gbias_1$',
                       '$gbias_2$',
                       '$gbias_3$',
                       '$gbias_4$',
                       '$uncorr_bias_1$',
                       '$uncorr_bias_2$',
                       '$uncorr_bias_3$',
                       '$uncorr_bias_4$',
                       '$uncorr_bias_5$',
                       '$m1$',
                       '$m2$',
                       '$m3$',
                       '$m4$',
                       '$a1_des$',
                       '$alpha1_des$',
                       '$a1_kids$',
                       '$alpha1_kids$',
                       '$logt_agn$',
                       '$omch2$',
                       '$ombh2$',
                       '$h0$',
                       '$n_s$',
                       '$s8_input$',
                       '$mnu$',
                       '$S_8$',
                       '$\\sigma_8$',
                       '$A_S$',
                       '$\Omega_m$',
                       '$OMEGA_LAMBDA$',
                       '$COSMOMC_THETA$',
                       '$BIAS_1$',
                       '$BIAS_2$',
                       '$BIAS_3$',
                       '$BIAS_4$',
                       '$BIAS_5$',
                       '$BIN_1$',
                       '$BIN_2$',
                       '$BIN_3$',
                       '$BIN_4$',
                       '$BIN_5$',
                       '$DELTA_Z_OUT_DES--BIN_1$',
                       '$DELTA_Z_OUT_DES--BIN_2$',
                       '$DELTA_Z_OUT_DES--BIN_3$',
                       '$DELTA_Z_OUT_DES--BIN_4$',
                       '$like_kids$',
                       '$like_des$',
                       '$like_norm$',
                       '$prior$',
                       '$like$',	
                       '$post$',	
                       '$weight$'
                ]

parameters_kids_shear = ['$\Omega_{cdm} \cdot h^2$',
                         '$\Omega_b \cdot h^2$',
                         "$h$",
                         "$n_s$",
                         "$S_8_input$",
                         "$halo-a$",
                         "$A_IA$",
                         'nofz_shifts--uncorr_bias_1',
                         'nofz_shifts--uncorr_bias_2',
                         'nofz_shifts--uncorr_bias_3',
                         'nofz_shifts--uncorr_bias_4',
                         'nofz_shifts--uncorr_bias_5',
                         '$S_8$',
                         '$\sigma_8$',
                         '$A_s$',
                         '$\Omega_m$',
                         '$\Omega_\nu$',
                         'COSMOLOGICAL_PARAMETERS--OMEGA_LAMBDA',
                         'COSMOLOGICAL_PARAMETERS--COSMOMC_THETA',
                         'NOFZ_SHIFTS--BIAS_1',
                         'NOFZ_SHIFTS--BIAS_2',
                         'NOFZ_SHIFTS--BIAS_3', 
                         'NOFZ_SHIFTS--BIAS_4', 
                         'NOFZ_SHIFTS--BIAS_5', 
                         'DELTA_Z_OUT--BIN_1', 
                         'DELTA_Z_OUT--BIN_2', 
                         'DELTA_Z_OUT--BIN_3',
                         'DELTA_Z_OUT--BIN_4',  
                         'DELTA_Z_OUT--BIN_5', 
                         'prior',  
                         'like', 
                         'post',  
                         '$weight$']

parameters_kids_3x2pt = ['$\Omega_{cdm} \cdot h^2$',
                         '$\Omega_b \cdot h^2$',
                         "$h$",
                         "$n_s$",
                         "$S_8_input$",
                         "$halo-a$",
                         "$A_IA$",      
                         "nofz_shifts--p_1", 
                         "nofz_shifts--p_2",  
                         "nofz_shifts--p_3", 
                         "nofz_shifts--p_4",
                         "nofz_shifts--p_5",  
                         "bias_parameters--b1_bin_1",
                         "bias_parameters--b2_bin_1",
                         "bias_parameters--gamma3_bin_1",
                         "bias_parameters--a_vir_bin_1",
                         "bias_parameters--b1_bin_2",
                         "bias_parameters--b2_bin_2",
                         "bias_parameters--gamma3_bin_2",
                         "bias_parameters--a_vir_bin_2",
                         '$S_8$',
                         '$\sigma_8$',
                         "COSMOLOGICAL_PARAMETERS--SIGMA_12",
                         "COSMOLOGICAL_PARAMETERS--A_S",
                         '$\Omega_m$',
                         "COSMOLOGICAL_PARAMETERS--OMEGA_NU",
                         "COSMOLOGICAL_PARAMETERS--OMEGA_LAMBDA",
                         "COSMOLOGICAL_PARAMETERS--COSMOMC_THETA",
                         "NOFZ_SHIFTS--BIAS_1", 
                         "NOFZ_SHIFTS--BIAS_2", 
                         "NOFZ_SHIFTS--BIAS_3", 
                         "NOFZ_SHIFTS--BIAS_4",
                         "NOFZ_SHIFTS--BIAS_5",  
                         "DELTA_Z_OUT--BIN_1",  
                         "DELTA_Z_OUT--BIN_2",  
                         "DELTA_Z_OUT--BIN_3",  
                         "DELTA_Z_OUT--BIN_4",  
                         "DELTA_Z_OUT--BIN_5",  
                         "prior",
                         "like", 
                         "post",
                         '$weight$'
]

fname = '/pscratch/sd/d/davidsan/Planck2018_chains/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE.paramnames'

# Initialize an empty dictionary named parameters_cmb
parameters_cmb = {}
parameters_cmb['weight'] = '$weight$'
parameters_cmb['like'] = '$like$'
# Read the file and store the parameter names in the dictionary
# First column is the key, second column is the value
with open(fname, 'r') as f:
    for line in f:
        # Split the line in the elements separated by \t
        elements = line.split('\t')
        key = elements[0]
        value = elements[1]
        # remove the newline character from the value
        value = f'${value[:-1]}$'
        # Store the key-value pair in the dictionary
        parameters_cmb[key] = value

parameters_hsc_y3_3x2pt_large = [
    "Ombh2",
    "Omch2",
    "Omde",
    "ln10p10As",
    "ns",
    "b1_0",
    "b1_1",
    "b1_2",
    "alphamag_0",
    "alphamag_1",
    "alphamag_2",
    "dm_0",
    "dpz_0",
    "AIA",
    "alphapsf",
    "betapsf",
    "$\Omega_m$",
    "$\sigma_8$",
    "$S_8$",
    "signal_dSigma0_0",
    "signal_dSigma0_1",
    "signal_dSigma0_2",
    "signal_dSigma0_3",
    "signal_dSigma1_0",
    "signal_dSigma1_1",
    "signal_dSigma1_2",
    "signal_dSigma1_3",
    "signal_dSigma1_4",
    "signal_dSigma2_0",
    "signal_dSigma2_1",
    "signal_dSigma2_2",
    "signal_dSigma2_3",
    "signal_dSigma2_4",
    "signal_dSigma2_5",
    "signal_dSigma2_6",
    "signal_dSigma2_7",
    "signal_xip_0",
    "signal_xip_1",
    "signal_xip_2",
    "signal_xip_3",
    "signal_xip_4",
    "signal_xip_5",
    "signal_xip_6",
    "signal_xip_7",
    "signal_xim_0",
    "signal_xim_1",
    "signal_xim_2",
    "signal_xim_3",
    "signal_xim_4",
    "signal_xim_5",
    "signal_xim_6",
    "signal_wp0_0",
    "signal_wp0_1",
    "signal_wp0_2",
    "signal_wp0_3",
    "signal_wp0_4",
    "signal_wp0_5",
    "signal_wp0_6",
    "signal_wp0_7",
    "signal_wp0_8",
    "signal_wp0_9",
    "signal_wp0_10",
    "signal_wp0_11",
    "signal_wp0_12",
    "signal_wp0_13",
    "signal_wp1_0",
    "signal_wp1_1",
    "signal_wp1_2",
    "signal_wp1_3",
    "signal_wp1_4",
    "signal_wp1_5",
    "signal_wp1_6",
    "signal_wp1_7",
    "signal_wp1_8",
    "signal_wp1_9",
    "signal_wp1_10",
    "signal_wp1_11",
    "signal_wp1_12",
    "signal_wp1_13",
    "signal_wp2_0",
    "signal_wp2_1",
    "signal_wp2_2",
    "signal_wp2_3",
    "signal_wp2_4",
    "signal_wp2_5",
    "signal_wp2_6",
    "signal_wp2_7",
    "signal_wp2_8",
    "signal_wp2_9",
    "signal_wp2_10",
    "signal_wp2_11",
    "signal_wp2_12",
    "signal_wp2_13",
    "lnlike",
    "lnpost"
]

parameters_hsc_y3_3x2pt_small = [
"Ombh2",
"Omch2",
"Omde",
"ln10p10As",
"ns",
"logMmin_0",
"logMmin_1",
"logMmin_2",
"sigma_sq_0",
"sigma_sq_1",
"sigma_sq_2",
"logM1_0",
"logM1_1",
"logM1_2",
"alpha_0",
"alpha_1",
"alpha_2",
"kappa_0",
"kappa_1",
"kappa_2",
"alphamag_0",
"alphamag_1",
"alphamag_2",
"dm_0",
"dpz_0",
"AIA",
"alphapsf",
"betapsf",
"$\Omega_m$",
"$\sigma_8$",
"$S_8$",
"signal_dSigma0_0",
"signal_dSigma0_1",
"signal_dSigma0_2",
"signal_dSigma0_3",
"signal_dSigma0_4",
"signal_dSigma0_5",
"signal_dSigma0_6",
"signal_dSigma0_7",
"signal_dSigma0_8",
"signal_dSigma1_0",
"signal_dSigma1_1",
"signal_dSigma1_2",
"signal_dSigma1_3",
"signal_dSigma1_4",
"signal_dSigma1_5",
"signal_dSigma1_6",
"signal_dSigma1_7",
"signal_dSigma1_8",
"signal_dSigma2_0",
"signal_dSigma2_1",
"signal_dSigma2_2",
"signal_dSigma2_3",
"signal_dSigma2_4",
"signal_dSigma2_5",
"signal_dSigma2_6",
"signal_dSigma2_7",
"signal_dSigma2_8",
"signal_xip_0",
"signal_xip_1",
"signal_xip_2",
"signal_xip_3",
"signal_xip_4",
"signal_xip_5",
"signal_xip_6",
"signal_xip_7",
"signal_xim_0",
"signal_xim_1",
"signal_xim_2",
"signal_xim_3",
"signal_xim_4",
"signal_xim_5",
"signal_xim_6",
"signal_wp0_0",
"signal_wp0_1",
"signal_wp0_2",
"signal_wp0_3",
"signal_wp0_4",
"signal_wp0_5",
"signal_wp0_6",
"signal_wp0_7",
"signal_wp0_8",
"signal_wp0_9",
"signal_wp0_10",
"signal_wp0_11",
"signal_wp0_12",
"signal_wp0_13",
"signal_wp0_14",
"signal_wp0_15",
"signal_wp1_0",
"signal_wp1_1",
"signal_wp1_2",
"signal_wp1_3",
"signal_wp1_4",
"signal_wp1_5",
"signal_wp1_6",
"signal_wp1_7",
"signal_wp1_8",
"signal_wp1_9",
"signal_wp1_10",
"signal_wp1_11",
"signal_wp1_12",
"signal_wp1_13",
"signal_wp1_14",
"signal_wp1_15",
"signal_wp2_0",
"signal_wp2_1",
"signal_wp2_2",
"signal_wp2_3",
"signal_wp2_4",
"signal_wp2_5",
"signal_wp2_6",
"signal_wp2_7",
"signal_wp2_8",
"signal_wp2_9",
"signal_wp2_10",
"signal_wp2_11",
"signal_wp2_12",
"signal_wp2_13",
"signal_wp2_14",
"signal_wp2_15",
"lnlike",
"lnpost",
]

########################################
###         Initialize chains        ###
########################################
def generate_hsc_chain(fname, chain_to_add):
    """
    Generate an HSC chainconsumer chain from a given file.

    Args:
        fname (str): Path to the txt chain file.
        chain_to_add (chainconsumer.Chain): The chainconsumer chain to add the HSC chain to.

    Returns:
        None

    Technical Details:
        This function loads the data from the specified file and extracts the weights and posterior values.
        It then adds the data to the existing chainconsumer chain, along with the specified parameters, weights,
        posterior, and kde values. The name of the chain is set to 'HSC Y1 1x2pt (Hikage et al.)'.
    """
    
    # HSC shear
    data_hsc = np.loadtxt(fname)
    weights_hsc = data_hsc[:, 0]
    posterior_hsc = data_hsc[:, 1]
    
    # Add to already initialized chain
    chain_to_add.add_chain(data_hsc, parameters=parameters_hsc,
                           weights=weights_hsc,
                           posterior=posterior_hsc,
                           kde=kde,
                           name='HSC Y1 1x2pt (Hikage et al.)')  # HSC shear Chain
    
    return()
def generate_hsc_chain(fname,chain_to_add):
    
    # fname - path to txt chain
    # Output - chainconsumer chain
    
    # HSC shear
    data_hsc = np.loadtxt(fname)
    weights_hsc = data_hsc[:,0]
    posterior_hsc = data_hsc[:,1]
    # data_hsc = data_hsc[:,-2:]
    
    # Add to already initialize chain
    chain_to_add.add_chain(data_hsc,parameters=parameters_hsc,
                            weights=weights_hsc,
                            posterior=posterior_hsc,
                            kde=kde,
                            name='HSC Y1 1x2pt (Hikage et al.)') # HSC shear Chain
    
    return()

def generate_hsc_chain_hamana(fname,chain_to_add):
    # fname - path to txt chain
    # Output - chainconsumer chain
    
    # HSC shear
    data_hsc = np.loadtxt(fname)
    weights_hsc = np.ones(data_hsc.shape[0])
    # posterior_hsc = data_hsc[:,1]
    
    # Add to already initialize chain
    chain_to_add.add_chain(data_hsc,
                           parameters=parameters_hsc_hamana,
                           weights=weights_hsc,
                           # posterior=posterior_hsc,
                           kde=kde,
                           name='HSC Y1 1x2pt (Hamana et al.)') # HSC shear Chain
    
    return()

def cosmosis_header(fname):
    # For a given chain file, reads the cosmosis header
    # and using a dictionary, translates from cosmosis
    # to latex and return parameters
    
    # Extract first line
    with open(fname) as file:
        header = file.readline()
    # Remove # 
    header = header[1:]
    # Splits elements \t
    parameters_cosmosis = header.split('\t')
    # Initialize empty list
    parameters_latex = list()
    # Obtain corresponding latex labels
    if 'hamana' in fname:
        for par in parameters_cosmosis:
            if par in parameters_dict_desc_hamana.keys():
                parameters_latex.append(parameters_dict_desc_hamana[par])
    else:
        for par in parameters_cosmosis:
            if par in parameters_dict_desc.keys():
                # print(par, 'in dictionary, adding to latex list: ', parameters_dict_desc[par])
                # print(parameters_dict_desc[par])
                parameters_latex.append(parameters_dict_desc[par])
            # else:
            #     print(par, 'not in dictionary')
    return(parameters_latex)

def omega_m(sample,parameters):
    # Compute Omega_m and add it to the sample
    if '$\\Omega_{cdm} \\cdot h^2$' in parameters and '$\\Omega_b \\cdot h^2$' in parameters:
        # Extract cdm, baryon and h
        omch2 = sample[:,parameters.index('$\\Omega_{cdm} \\cdot h^2$')]
        # print(parameters.index('$\\Omega_{cdm} \\cdot h^2$'))
        ombh2 = sample[:,parameters.index('$\\Omega_b \\cdot h^2$')]
        # print(parameters.index('$\\Omega_b \\cdot h^2$'))
        h = sample[:,parameters.index('$h$')]
        # print(parameters.index('$h$'))
        # Compute omega_m
        om = (omch2 + ombh2) / h ** 2
    elif '$\\Omega_{cdm}$' in parameters and '$\\Omega_b$' in parameters:
        # Extract cdm, baryon and h
        omc = sample[:,parameters.index('$\Omega_{cdm}$')]
        # print(parameters.index('$\\Omega_{cdm} \\cdot h^2$'))
        omb = sample[:,parameters.index('$\Omega_b$')]
        # Compute omega_m
        om = omc + omb
    elif '$\\Omega_m$' in parameters:
        om = sample[:,parameters.index('$\\Omega_m$')]
    else:
        print('Invalid case')
    # print(np.mean(om))
    # Add column to the main sample
    sample = np.c_[sample,om]
    parameters.append('$\\Omega_m$')
    
    return(sample,parameters)

def S8(sample,parameters,alpha=0.5):
    # Compute S8
    
    # Extract om and sigma8
    om=sample[:,parameters.index('$\\Omega_m$')]
    """ plt.hist(om, histtype='step')
    plt.title('$\Omega_m$')
    plt.show()
    plt.close()
    print(parameters.index('$\\Omega_m$'))
    print('om',np.mean(om)) """
    sigma8=sample[:,parameters.index('$\\sigma_8$')]
    """ plt.hist(sigma8, histtype='step')
    plt.title('$\sigma_8$')
    plt.show()
    plt.close()
    print(parameters.index('$\\sigma_8$'))
    print('sigma8',np.mean(sigma8)) """
    # Compute S8
    S8=sigma8*(om/0.3)**alpha
    """ plt.hist(S8, histtype='step')
    plt.title('$S_8$')
    plt.show()
    plt.close()
    print('S8',np.mean(S8)) """
    # Find the slope of the correlation between om and sigma8
    slope, intercept, r_value, p_value, std_err = stats.linregress(om,sigma8)
    # Print the slope info
    print(f'Slope Om-sigma8: {slope}')
    # Add column to the main sample
    sample = np.c_[sample,S8]
    parameters.append('$S_8$')
    
    return(sample,parameters)

def generate_cosmosis_chain(fname_list,
                            chain_label_list,
                            add_planck = False,
                            add_hsc_hikage = False,
                            add_hsc_hamana = False,
                            add_nicola = False,
                            add_prior = True,
                            S8_alpha = 0.5,
                            show_auxplots = False,
                            burnin = True, 
                            trace_plots = False):
    # fname - path to txt chain (could be a list of chains to compare)
    # add_hsc add comparison with HSC contours

    print('Generating chainconsumer chain')
    print('>> Default alpha for S8: ', S8_alpha, ',   S8 = sigma8*(om/0.3)**alpha')
    
    # Output - chainconsumer chain
    fname_hsc = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/hsc_chains/using_cls/HSC_Y1_LCDM_post_fid.txt')
    fname_hsc_hamana = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/hsc_chains/using_corr_function/HSC_hamana2020_fiducial/hsc_hamana2020_fiducial.txt')
    fname_nicola = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/carlos_chains/hsc_dz_Andrina_ell300/hsc_dz_Andrina_ell300.merged.txt')
    fname_planck = os.path.join('/pscratch/sd/d/davidsan/Planck2018_chains/base/plikHM_TTTEEE_lowl_lowEbase_plikHM_TTTEEE_lowl_lowE.txt')
    # Initialize empty chainconsumer object
    c = ChainConsumer()
    
    if add_hsc_hikage == True:
        fname_list = np.append(fname_list,fname_hsc)
        chain_label_list = np.append(chain_label_list,'1x2pt HSC Year 1 (Hikage et al.)')
    if add_hsc_hamana == True:
        fname_list = np.append(fname_list,fname_hsc_hamana)
        chain_label_list = np.append(chain_label_list,'1x2pt HSC Year 1 (Hamana et al.)')
    if add_nicola == True:
        fname_list = np.append(fname_list,fname_nicola)
        chain_label_list = np.append(chain_label_list,'1x2pt Nicola et al.')
    if add_planck == True:
        fname_list = np.append(fname_list,fname_planck)
    if add_prior == True:
        parameters_priors=[# Cosmology
                           '$\Omega_{cdm} \cdot h^2$',
                           '$\Omega_b \cdot h^2$',
                           '$h$',
                           '$\ln(10^{10} A_s)$',
                           '$n_{s}$',
                            # Lens photo-z uncert.
                           '$\Delta z^{lens}_1$',
                           '$\Delta z^{lens}_2$',
                           '$\Delta z^{lens}_3$',
                           '$\Delta z^{lens}_4$',
                           '$\sigma z^{lens}_1$',
                           '$\sigma z^{lens}_2$',
                           '$\sigma z^{lens}_3$',
                           '$\sigma z^{lens}_4$',
                            # Galaxy bias
                           '$b^{lens}_1$',
                           '$b^{lens}_2$',
                           '$b^{lens}_3$',
                           '$b^{lens}_4$',
                            # Sources photo-z uncert.
                           '$\Delta z^{source}_1$',
                           '$\Delta z^{source}_2$',
                           '$\Delta z^{source}_3$',
                           '$\Delta z^{source}_4$',
                           '$\sigma z^{source}_1$',
                           '$\sigma z^{source}_2$',
                           '$\sigma z^{source}_3$',
                           '$\sigma z^{source}_4$',
                            # Multiplicative shear bias
                           'm',
                           # Non-linear intrinsic alignment
                           '$A_{IA}$',
                           '$\eta$',
                           # Linear Intrinsic Alignment
                           '$A_{IA,lin}$',
                           '$\\alpha_z$']

        priors = np.c_[# Cosmology
                       uniform(0.03,0.7,size=10000000),
                       uniform(0.019,0.026,size=10000000),
                       uniform(0.6,0.9,size=10000000),
                       uniform(1.5,6.0,size=10000000),
                       uniform(0.87,1.07,size=10000000),
                        # Lens photo-z uncert.
                       normal(0,0.0285,size=10000000),
                       normal(0,0.0135,size=10000000),
                       normal(0,0.0383,size=10000000),
                       normal(0,0.0376,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                        # Galaxy bias
                       uniform(0.8,3.0,size=10000000),
                       uniform(0.8,3.0,size=10000000),
                       uniform(0.8,3.0,size=10000000),
                       uniform(0.8,3.0,size=10000000),
                        # Sources photo-z uncert.
                       normal(0,0.0285,size=10000000),
                       normal(0,0.0135,size=10000000),
                       normal(0,0.0383,size=10000000),
                       normal(0,0.0376,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                       normal(1.0,0.05,size=10000000),
                        # Multiplicative shear bias
                       normal(0.0,0.01,size=10000000),
                       # Non-linear intrinsic alignment
                       uniform(-5.0,5.0,size=10000000),
                       uniform(-5.0,5.0,size=10000000),
                       # Linear Intrinsic Alignment
                       uniform(-5.0,5.0,size=10000000),
                       uniform(-5.0,5.0,size=10000000)]

        c.add_chain(priors,parameters=parameters_priors,name='Prior',show_as_1d_prior=True)
            
    # print(type(fname_list))
    # Chain index to choose the color
    k = 0
    for fname,chain_label in zip(fname_list,chain_label_list):
        print(f'Adding ... {chain_label}')
        if fname == fname_hsc:
            # Read Hikage et al. HSC official chain
            sample = np.loadtxt(fname)
            # All weights are 1 
            weights = sample[:,0]
            if show_auxplots == True:
                # Weights checking
                plt.plot(np.arange(len(weights)), weights)
                plt.show()
                plt.close()
            # Extract the posterior
            posterior = sample[:,1]
            # Read the parameters
            parameters = list(np.copy(parameters_hsc))
            # Appending S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=S8_alpha)
            # S8_hikage = sample[:,parameters.index('$S_8$')]
            if show_auxplots:
                fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(9, 1.5))
                for ind, par in enumerate(['$\Omega_{cdm} \cdot h^2$', '$\Omega_b \cdot h^2$', '$\Omega_m$', '$\sigma_8$', '$S_8$', '$\ln(10^{10} A_s)$']):
                    axs[ind].hist(sample[:,parameters.index(par)],density=True,histtype='step')
                    axs[ind].set_title(par)
                plt.show()
                plt.close()
            # Re-scaling multiplicative shear bias 100 * Delta m
            col = parameters.index('m')
            sample[:,col] /= 100
            # sigmas = np.array([2.85, 1.35, 3.83, 3.76])/100
            col = parameters.index('$\Delta z^{source}_1$')
            sigma = 2.85 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_2$')
            sigma = 1.35 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_3$')
            sigma = 3.83 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_4$')
            sigma = 3.76 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weights,
                        posterior=posterior,
                        kde=kde,
                        name=chain_label)
        elif fname == fname_nicola:
            sample = np.loadtxt(fname)
            weights = sample[:,0]
            posterior = sample[:,1]
            parameters = list(np.copy(parameters_nicola))
            # Sampling Om_c and Om_b
            sample,parameters=omega_m(sample=sample,parameters=parameters)
            # Appendin S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=S8_alpha)

            if show_auxplots == True:
                fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(9, 1.5))

                for i, par in enumerate(['$\Omega_{cdm}$', '$\Omega_b$', '$\Omega_m$', '$\sigma_8$', '$S_8$']):
                    axs[i].hist(sample[:,parameters.index(par)],density=True,histtype='step')
                    axs[i].set_title(par)
                
                plt.show()
                plt.close()
            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weights,
                        # posterior=posterior,
                        kde=kde,
                        name=chain_label)
            
        elif fname == fname_hsc_hamana:
            sample = np.loadtxt(fname)
            weights = sample[:,0]
            if show_auxplots == True:
                # Weights checking
                plt.plot(np.arange(len(weights)),weights)
                plt.show()
                plt.close()
            # posterior = sample[:,1]
            parameters = list(np.copy(parameters_hsc_hamana))
            # Re-scaling multiplicative shear bias 100 * Delta m
            col = parameters.index('m')
            sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_1$')
            sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_2$')
            sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_3$')
            sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_4$')
            sample[:,col] /= 100
            col = parameters.index('$\Omega_b$')
            sample[:,col] /= 100
            col = parameters.index('$h$')
            sample[:,col] /= 100

            if show_auxplots == True:
                # Histograms
                fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(9, 1.5))
                
                for ind, par in enumerate(['$\Omega_{cdm}$', '$\Omega_b$', '$\Omega_m$', '$\sigma_8$', '$S_8$', '$\ln(10^{9} A_s)$']):
                    axs[ind].hist(sample[:,parameters.index(par)],density=True,histtype='step')
                    axs[ind].set_title(par)
                
                plt.show()
                plt.close()
            
            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weights,
                        # posterior=posterior,
                        kde=kde,
                        name=chain_label)
        else:
            if 'post_equal_weights.dat' in fname:
                print('>> Loading equal weights chain')
                sample = np.loadtxt(fname)
                weights = np.ones(sample.shape[0])
                posterior = sample[:,-1]
                # Extract the parameters
                if 'hikage' in fname:
                    print('>> Hikage-like chain')
                    
                    if 'no_syst' in fname:
                        print('>> >> no modeling syst.')
                        parameters = list(np.copy(parameters_mn_eqwpost_no_syst))
                    else:
                        parameters = list(np.copy(parameters_mn_eqwpost))
                        
                elif 'hamana' in fname:
                    print('>> Hamana-like chain')
                    
                    if 'no_syst' in fname:
                        parameters = list(np.copy(parameters_mn_eqwpost_hamana_no_syst))
                    else:
                        parameters = list(np.copy(parameters_mn_eqwpost_hamana))
                        
                elif 'txpipe' in fname:
                    print('>> This work')
                    parameters = list(np.copy(parameters_mn_eqwpost))

                
                else:
                    print('>> Reading parameters')
                    parameters = list(np.copy(parameters_mn_eqwpost_new))
                
                if '$\Omega_m$' in parameters:
                    print('>> Omega_m already computed during sampling')
                else: 
                    # Sampling Om_c*h^2 and Om_b*h^2
                    sample,parameters=omega_m(sample=sample,parameters=parameters)
                if '$S_8$' not in parameters:
                    print('>> Computing S8')
                    # Computing S_8
                    sample,parameters=S8(sample=sample,parameters=parameters,alpha=S8_alpha)
                else:
                    print('>> S8 already derived during sampling')
                # print(sample,parameters)
            else:
                print('>> Chain with weights')
                # Our MCMC
                sample = np.loadtxt(fname)
                if 'DES Year 3 x KiDS 1000' in chain_label:
                    parameters = list(np.copy(parameters_des_kids))
                elif 'DES' in chain_label and '3x2pt' in chain_label:
                    parameters = list(np.copy(parameters_des_3x2pt))
                elif 'KiDS-1000 - 3x2pt' in chain_label:
                    parameters = list(np.copy(parameters_kids_3x2pt))
                elif 'HSC Year 3 - 3x2pt Small scales' in chain_label:
                    parameters = list(np.copy(parameters_hsc_y3_3x2pt_small))
                elif 'DES Year 3 x KiDS 1000 - 1x2pt' in chain_label:
                    parameters = list(np.copy(parameters_des_kids))
                elif 'Planck 2018 TTTEEE' in chain_label:
                    parameters = list(parameters_cmb.values())
                
                
                
                
                
                
                
                
                
                
                
                else: 
                    # Extract parameters name from chain
                    parameters = cosmosis_header(fname=fname)
                # Extract weights and posterior
                if '$weight$' in parameters:
                    weights = sample[:,parameters.index('$weight$')]
                else:
                    weights = np.ones(sample.shape[0])
                # Burn-in cut
                # Find the first index where the weight is larger than 1e-6
                ind_burnin = np.argmax(weights>1e-6)
                print(f'>> Burn-in cut at index {ind_burnin}')
                if show_auxplots == True:
                    # Plot the weights
                    plt.plot(np.arange(len(weights)),weights, color='k')
                    # Plot a dashed vertical line at that index
                    plt.axvline(x=ind_burnin, linestyle='--', color='r')
                    plt.show()
                    plt.close()
                if burnin == True:
                    # Apply burn-in cut
                    print(f'>> Chain size before burn-in cut: {sample.shape}')
                    sample = sample[ind_burnin:,:]
                    print(f'>> Chain size after burn-in cut: {sample.shape}')
                # Extract the posterior
                if '$post$' in parameters:
                    posterior = sample[:,parameters.index('$post$')] 
                else: 
                    posterior = np.ones(sample.shape[0])
                if '$weight$' in parameters:
                    weights = sample[:,parameters.index('$weight$')]
                else:
                    weights = np.ones(sample.shape[0])
                if '$\Omega_m$' in parameters:
                    print('>> Omega_m already computed during sampling')
                else:
                    print('>> Computing Omega_m from sampled Om_c*h^2 and Om_b*h^2')
                    # Sampling Om_c*h^2 and Om_b*h^2
                    sample,parameters=omega_m(sample=sample,parameters=parameters)
                if '$S_8$' not in parameters:
                    print('>> Computing S8')
                    # Computing S_8
                    sample,parameters=S8(sample=sample,parameters=parameters,alpha=S8_alpha) 
                else:
                    print('>> S8 already computed')  

                if trace_plots == True:
                    for par_trace in ['$\Omega_{cdm} \cdot h^2$','$\Omega_b \cdot h^2$','$\Omega_m$','$\sigma_8$','$S_8$']:
                        plt.plot(np.arange(len(sample[:,parameters.index(par_trace)])),sample[:,parameters.index(par_trace)])
                        plt.title(par_trace)
                        plt.show()
                        plt.close()

            if show_auxplots == True:
                # Histograms
                params_to_plot = np.array([])
                for par in parameters:
                    if par in ['$\Omega_{cdm} \cdot h^2$','$\Omega_{cdm}','$\Omega_b \cdot h^2$','$\Omega_b$', \
                            '$\Omega_m$','$\ln(10^{9} A_s)$','$\ln(10^{10} A_s)$','$\sigma_8$','$S_8$']:
                        params_to_plot = np.append(params_to_plot,par)
                fig, axs = plt.subplots(nrows=1, ncols=int(len(params_to_plot)), figsize=(9, 1.5))

                for ind, par in enumerate(params_to_plot):
                    axs[ind].hist(sample[:,parameters.index(par)],density=True,histtype='step')
                    axs[ind].set_title(par)

                plt.show()
                plt.close()

            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weights,
                        posterior=posterior,
                        kde=kde,
                        name=chain_label)

        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\Omega_m$')], weights = weights)
        print(f'Omega_matter = {med}+{up}-{lo}')
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\sigma_8$')], weights = weights)
        print(f'sigma_8 = {med}+{up}-{lo}')
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$S_8$')], weights = weights)
        print(f'S8 = {med}+{up}-{lo}')
        # if report_median == True:
        # for par in parameters:
        #     med,up,lo=report_median_68assy(arr=sample[:,parameters.index(par)])
        #     print(f'{par} = {med}+{up}-{lo}')
        print("-------------------------------")
        k += 1
    # Config chain
    c.configure(flip=False,
                # linestyles=["-"]*len(fname_list),
                linewidths=[1.2]*len(fname_list),
                shade=[False]+[False]*len(fname_list),
                legend_kwargs={"fontsize": 8},#, "loc": "upper right"},
                #legend_location=(0, 0),
                watermark_text_kwargs={"alpha": 0.2,"weight": "bold"},
                colors=colors,
                max_ticks=4,
                serif=True)
    return(c)
###############################
### Some analysis functions ###
###############################
def sigma_68(arr,axis=None):
    upper,lower=np.percentile(arr,[84.075,15.825],axis=axis)
    return(upper, lower)

def weighted_median(data, weights):
    """
    Compute the weighted median of data.
    
    Parameters:
    data : list or numpy array
        Data points.
    weights : list or numpy array
        Weights corresponding to data points.
        
    Returns:
    weighted_median : float
        The weighted median.
    """
    # Combine the data and weights and sort by data
    combined = list(zip(data, weights))
    combined.sort(key=lambda x: x[0])
    data_sorted, weights_sorted = zip(*combined)
    
    # Compute the cumulative weight
    cumulative_weights = [sum(weights_sorted[:i+1]) for i in range(len(weights_sorted))]
    half_total_weight = sum(weights_sorted) / 2
    
    # Find the weighted median
    for i, weight in enumerate(cumulative_weights):
        if weight >= half_total_weight:
            return data_sorted[i]


def report_median_68assy(arr, weights):
    # arr = data_hsc[:,parameters_hsc.index('$\Omega_m$')]
    # median = np.median(arr)
    median = weighted_median(data = arr, weights = weights)
    upper, lower = sigma_68(arr, axis=None)
    upper = upper - median
    lower = median - lower 
    median = np.round(median, 3)
    upper = np.round(upper, 3)
    lower = np.round(lower, 3)
    # print(f'{median}+{upper}-{lower}')
    return(median, upper, lower)

def report_mean_68assy(arr):
    # arr = data_hsc[:,parameters_hsc.index('$\Omega_m$')]
    mean = np.mean(arr)
    upper, lower = sigma_68(arr, axis=None)
    upper = upper - mean
    lower = mean - lower 
    median = np.round(mean, 3)
    upper = np.round(upper, 3)
    lower = np.round(lower, 3)
    # print(f'{median}+{upper}-{lower}')
    return(mean, upper, lower)

def report_best_fit(filename,labeltxt,paramslist,path_save='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/bestfit_files/clustering'):
    # Goal - report median and 68% CL
    
    # Read data
    data_desc=np.loadtxt(filename)
    
    # Generate Omegam (using current order of parameters)
    omega_m=(data_desc[:,0]+data_desc[:,1])/data_desc[:,2]
    data_desc=np.c_[omega_m,data_desc]
    
    # Generate S8 (using current order of parameters)
    S8=data_desc[:,-6]*np.sqrt(data_desc[:,0]/0.3)
    data_desc=np.c_[S8,data_desc]
    
    with open(os.path.join(path_save,f'{labeltxt}.txt'),'w') as file:
        for i,el in enumerate(paramslist):
            file.write(f'{el}={report_median_68assy(data_desc[:,i])[0]}+{report_median_68assy(data_desc[:,i])[1]}-{report_median_68assy(data_desc[:,i])[2]} \n')
            
    file.close()
    return()

##########################
### Plotting functions ###
##########################
def plot_S8(chain,labelpng,S8_alpha,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot_distributions(parameters=['$S_8$'], 
                             extents=extents_dict,
                             figsize=(3,3))
    # Add text in the top right corner showing the alpha value in relative position 
    # to the current axis
    fig.text(0.2, 0.8, r'$\alpha$ = '+str(np.round(S8_alpha, 2)), fontsize=8)
    plt.savefig(os.path.join(savepath,f'S8_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_sigma8_S8(chain,labelpng,S8_alpha,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    chain.configure(
                legend_kwargs={"fontsize": 12},
                legend_location=(0, 0),
                )
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$S_8$'], 
                             extents=extents_dict,
                             # watermark=r"Preliminary",
                             figsize=(5,5))
    # Add text in the last panel with the S8 alpha value
    # fig.text(0.2, 0.8, r'$\alpha$ = '+str(np.round(S8_alpha, 2)), fontsize=8)
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'Om_sigma8_S8_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'Om_sigma8_S8_{labelpng}.pdf'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_sigma8_S8_w(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$S_8$', 'w'], 
                             extents=extents_dict,
                             # watermark=r"Preliminary",
                             figsize=(5,5))
    # Add text in the last panel with the S8 alpha value
    # fig.text(0.2, 0.8, r'$\alpha$ = '+str(np.round(S8_alpha, 2)), fontsize=8)
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'Om_sigma8_S8_w_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'Om_sigma8_S8_w_{labelpng}.pdf'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_sigma8_lnAs(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$\ln(10^{10} A_s)$'], 
                             extents=extents_dict,
                             # watermark=r"Preliminary",
                             figsize=(3,3)) 
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'Om_sigma8_lnAs_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'Om_sigma8_lnAs_{labelpng}.pdf'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_sigma8_Hikage(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    # chain.configure(plot_hists=False,
    #                 legend_kwargs={"fontsize": 6})
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$'], 
                             extents=extents_dict,
                             figsize=(2,2))
    plt.savefig(os.path.join(savepath,f'Om_sigma8_Hikage_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)


def plot_Omegam_sigma8(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$'], 
                             extents=extents_dict)
                             # watermark="Preliminary")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'Om_sigma8_{labelpng}.png'),
                dpi=300,
                bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'Om_sigma8_{labelpng}.pdf'),
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_S8(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$S_8$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(3,3))
    plt.savefig(os.path.join(savepath,f'Om_S8_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\sigma_8$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(8,8))
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.pdf'),
                dpi=300,
                bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.png'),
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_hamana(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\sigma_8$','$\ln(10^{9} A_s)$', '$\Omega_b$','$\Omega_{cdm}$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(3,3))
    plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_as(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\ln(10^{10} A_s)$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(3,3))
    plt.savefig(os.path.join(savepath,f'cosmo_as_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_mnu(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\sigma_8$', '$S_8$', '$\\Omega_\\nu h^2$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$', '$h$', '$n_{s}$', '$\ln(10^{10} A_s)$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="GROW")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'cosmo_mnu_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_def(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_{cdm} \cdot h^2$','$\Omega_b \cdot h^2$','$h$','$n_{s}$','$\ln(10^{10} A_s)$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(3,3))
    plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###     Lens photo-z uncert.       ###
######################################
def plot_pz_deltaz_lens(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Delta z^{lens}_1$','$\Delta z^{lens}_2$','$\Delta z^{lens}_3$','$\Delta z^{lens}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="column")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'delta_pz_lens_{labelpng}.png'),
                dpi=300,
                bbox_inches='tight')
    
        plt.savefig(os.path.join(savepath,f'delta_pz_lens_{labelpng}.pdf'),
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_pz_stretch_lens(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\sigma z^{lens}_1$','$\sigma z^{lens}_2$','$\sigma z^{lens}_3$','$\sigma z^{lens}_4$'],
                             extents=extents_dict,
                             #watermark="Preliminary",
                             figsize="column")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'stretch_pz_lens_{labelpng}.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'stretch_pz_lens_{labelpng}.pdf'),
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_consistency_lenses(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$S_8$',
                                         '$b^{lens}_1$','$b^{lens}_2$','$b^{lens}_3$','$b^{lens}_4$',
                                         '$\Delta z^{lens}_1$','$\Delta z^{lens}_2$','$\Delta z^{lens}_3$','$\Delta z^{lens}_4$',
                                         '$\sigma z^{lens}_1$','$\sigma z^{lens}_2$','$\sigma z^{lens}_3$','$\sigma z^{lens}_4$'],
                             extents=extents_dict,
                             figsize="GROW")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'consistency_lenses_{labelpng}.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'consistency_lenses_{labelpng}.pdf'),
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###   Sources photo-z uncert.      ###
######################################
def plot_pz_deltaz_source(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Delta z^{source}_1$','$\Delta z^{source}_2$','$\Delta z^{source}_3$','$\Delta z^{source}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="column")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'delta_pz_source_{labelpng}.png'),
                dpi=300,
                bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'delta_pz_source_{labelpng}.pdf'),
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_pz_stretch_source(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\sigma z^{source}_1$','$\sigma z^{source}_2$','$\sigma z^{source}_3$','$\sigma z^{source}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="column")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'stretch_pz_source_{labelpng}.png'),
                dpi=300,
                bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'stretch_pz_source_{labelpng}.pdf'),
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###          Galaxy bias           ###
######################################
def plot_galbias_lens(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    chain.configure(
                legend_kwargs={"fontsize": 8},
                legend_location=(0, 0),
                )
    fig = chain.plotter.plot(parameters=['$b^{lens}_1$','$b^{lens}_2$','$b^{lens}_3$','$b^{lens}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="column")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'galbias_lens_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'galbias_lens_{labelpng}.pdf'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_intrinsic_alignments(chain,labelpng,mode="lin",savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/shear'):
    # mode = "lin" or "nla" for linear o non-linear alignment, respectively
    chain.configure(
                legend_kwargs={"fontsize": 8},
                legend_location=(0, 0),
                )
    if mode == "nla":
        fig = chain.plotter.plot(parameters=['$A_{IA}$','$\eta$'],
                                 extents=extents_dict,
                                 # watermark="Preliminary",
                                 figsize=(6,6))
        if savepath is not None:
            plt.savefig(os.path.join(savepath,f'nla-ia_{labelpng}.png'),
                       dpi=300,
                       bbox_inches='tight')
            plt.savefig(os.path.join(savepath,f'nla-ia_{labelpng}.pdf'),
                       dpi=300,
                       bbox_inches='tight')
    elif mode == "lin":
        fig = chain.plotter.plot(parameters=['$A_{IA,lin}$','$\\alpha_z$'],
                                 extents=extents_dict,
                                 # watermark="Preliminary",
                                 figsize=(3,3))
        if savepath is not None:
            plt.savefig(os.path.join(savepath,f'lin-ia_{labelpng}.png'),
                       dpi=300,
                       bbox_inches='tight')
            plt.savefig(os.path.join(savepath,f'lin-ia_{labelpng}.pdf'),
                       dpi=300,
                       bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###  Multiplicative shear bias     ###
######################################
def plot_delta_m(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['m'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize=(3,3))
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'delta_m_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###      Full-triangle plot        ###
######################################

def plot_full_shear_hikage(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    chain.configure(
                    legend_kwargs={"fontsize": 16}
                   )
    fig = chain.plotter.plot(parameters=[# Cosmology
                                        '$\Omega_m$','$\sigma_8$','$S_8$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$','$n_{s}$','$h$','$\ln(10^{10} A_s)$',
                                         # Intrinsic alignment
                                        '$A_{IA}$','$\eta$',
                                         # multiplicative shear bias
                                        'm',
                                         # photo-z
                                         '$\Delta z^{source}_1$','$\Delta z^{source}_2$','$\Delta z^{source}_3$','$\Delta z^{source}_4$'],
                                         # '$\sigma z^{source}_1$','$\sigma z^{source}_2$','$\sigma z^{source}_3$','$\sigma z^{source}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="GROW")
    if savepath is not None:
        plt.savefig(os.path.join(savepath,f'full_shear_hikage_{labelpng}.png'),
                dpi=500,
                bbox_inches='tight')
        plt.savefig(os.path.join(savepath,f'full_shear_hikage_{labelpng}.pdf'),
                dpi=500,
                bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_full_shear_hamana(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=[# Cosmology
                                        '$\Omega_m$','$\sigma_8$','$S_8$','$\Omega_{cdm}$','$\Omega_b$','$n_{s}$','$h$','$\ln(10^{9} A_s)$',
                                         # Intrinsic alignment
                                        '$A_{IA}$','$\eta$',
                                         # multiplicative shear bias
                                        'm',
                                         # photo-z
                                         '$\Delta z^{source}_1$','$\Delta z^{source}_2$','$\Delta z^{source}_3$','$\Delta z^{source}_4$'],
                             extents=extents_dict,
                             # watermark="Preliminary",
                             figsize="GROW")
    plt.savefig(os.path.join(savepath,f'full_shear_hamana_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def S8_comparison_plotter(chain_list, label_list, bold_indices, labelpng, figsize):
    """
    Plots a comparison of S8, Omega_m, and sigma8 values for different chains.

    Args:
        chain_list (list): List of file paths to the chains.
        label_list (list): List of labels for each chain.
        labelpng (str): Label for the output PNG file.

    Returns:
        None
    """
    from operator import index
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.special import ndtri

    # Create a figure
    # figsize = (25, 10)
    fig = plt.figure(figsize=figsize)

    # Define the GridSpec
    gs = gridspec.GridSpec(1, 3, figure=fig)  # 1 row, 4 columns

    # Add subplots
    ax1 = fig.add_subplot(gs[0, 0])  # First subplot spans columns 0 and 1
    ax1.set_xlabel('$S_8$', fontsize=50)
    ax2 = fig.add_subplot(gs[0, 1])    # Second subplot in column 2
    ax2.set_xlabel('$\Omega_m$', fontsize=50)
    ax3 = fig.add_subplot(gs[0, 2])    # Third subplot in column 3
    ax3.set_xlabel('$\sigma_8$', fontsize=50)

    # Increase size of xticks
    ax1.tick_params(axis='x', labelsize=30)
    ax2.tick_params(axis='x', labelsize=30)
    ax3.tick_params(axis='x', labelsize=30)

    # Remove y ticks from all panels
    # ax1.set_yticks([])
    ax1.set_ylim([-2,len(label_list)])
    ax2.set_yticks([])
    ax2.set_ylim([-2,len(label_list)])
    ax3.set_yticks([])
    ax3.set_ylim([-2,len(label_list)])

    index = 0
    color_array = []
    row_index = np.arange(len(label_list))[::-1]
    for label, fname in zip(label_list,chain_list):
        # print('##################################################################33')
        # print('Chain: ', index + 1, label)
        if fname == '/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/hsc_chains/using_cls/HSC_Y1_LCDM_post_fid.txt':
            print('Hikage et al. cosmic shear')
            # Read Hikage et al. HSC official chain
            sample = np.loadtxt(fname)
            # All weights are 1 
            weights = sample[:,0]
            # Read the parameters
            parameters = list(np.copy(parameters_hsc))
            # Appending S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=0.5)
            # Re-scaling multiplicative shear bias 100 * Delta m
            col = parameters.index('m')
            sample[:,col] /= 100
            # sigmas = np.array([2.85, 1.35, 3.83, 3.76])/100
            col = parameters.index('$\Delta z^{source}_1$')
            sigma = 2.85 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_2$')
            sigma = 1.35 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_3$')
            sigma = 3.83 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
            # sample[:,col] /= 100
            col = parameters.index('$\Delta z^{source}_4$')
            sigma = 3.76 / 100
            sample[:,col] = ndtri(sample[:,col]) * sigma
        elif fname == '/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/carlos_chains/hsc_dz_Andrina_ell300/hsc_dz_Andrina_ell300.merged.txt':
            print('Nicola et al. / Garcia-Garcia et al. HSC Year 1 Cl')
            sample = np.loadtxt(fname)
            weights = sample[:,0]
            posterior = sample[:,1]
            parameters = list(np.copy(parameters_nicola))
            # Sampling Om_c and Om_b
            sample,parameters=omega_m(sample=sample,parameters=parameters)
            # Appendin S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=0.5)
        elif fname == '/pscratch/sd/d/davidsan/Planck2018_chains/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE.txt':
            print('Planck 2018 TTTEEE')
            sample = np.loadtxt(fname)
            parameters = list(parameters_cmb.values())
            weights = sample[:,parameters.index('$weight$')]
            # Normalize the area
        elif fname == '/pscratch/sd/d/davidsan/KiDS1000_vdB22_cosmic_shear_data_release/multinest/Fid_output_multinest_C.txt':
            print('KiDS 1000 cosmic shear')
            sample = np.loadtxt(fname)
            parameters = list(np.copy(parameters_kids_shear))
            weights = sample[:, parameters.index('$weight$')]
        elif fname == '/pscratch/sd/d/davidsan/KiDS1000_3x2pt_fiducial_chains/cosmology/samples_multinest_blindC_EE_nE_w.txt':
            print('KiDS 1000 3x2pt')
            sample = np.loadtxt(fname)
            parameters = list(np.copy(parameters_kids_3x2pt))
            weights = sample[:, parameters.index('$weight$')]
        elif fname == '/pscratch/sd/d/davidsan/HSC_Year_3_chains/final_y3_chains/hsc_y3_3x2pt_large_scale.txt':
            print('HSC Year 3 3x2pt Large scales')
            sample = np.loadtxt(fname)
            parameters = list(np.copy(parameters_hsc_y3_3x2pt_large))
            weights = np.ones(sample.shape[0])
        elif fname == '/pscratch/sd/d/davidsan/HSC_Year_3_chains/final_y3_chains/hsc_y3_3x2pt_small_scale.txt':
            print('HSC Year 3 3x2pt Small scales')
            sample = np.loadtxt(fname)
            parameters = list(np.copy(parameters_hsc_y3_3x2pt_small))
            weights = np.ones(sample.shape[0])
        else:   
            sample = np.loadtxt(fname)
            if 'DES Year 3 x KiDS 1000' in label:
                parameters = list(np.copy(parameters_des_kids))
            elif 'DES' in label and '3x2pt' in label:
                parameters = list(np.copy(parameters_des_3x2pt))
            elif 'DES' in label and '2x2pt' in label:
                parameters = list(np.copy(parameters_des_2x2pt))
            elif 'DES' in label and '1x2pt' in label :
                parameters = list(np.copy(parameters_des_1x2pt))
            else:
                # Extract parameters name from chain
                parameters = cosmosis_header(fname=fname)
                # print(parameters)
            if ('$weight$' in parameters) and ('Planck' not in label):
                # Extract weights and posterior
                weights = sample[:,parameters.index('$weight$')]
                # Burn-in cut
                # Find the first index where the weight is larger than 1e-6
                ind_burnin = np.argmax(weights>1e-6)
                # Apply burn-in cut
                sample = sample[ind_burnin:,:]
                # Extract weights and posterior
                weights = sample[:,parameters.index('$weight$')]
            if '$\Omega_m$' in parameters:
                print('>> Omega_m already computed during sampling')
            else:
                print('>> Computing Omega_m from sampled Om_c*h^2 and Om_b*h^2')
                # Sampling Om_c*h^2 and Om_b*h^2
                sample,parameters=omega_m(sample=sample,parameters=parameters)
            if '$S_8$' not in parameters:
                print('>> Computing S8')
                # Computing S_8
                sample,parameters=S8(sample=sample,parameters=parameters,alpha=0.5) 
            else:
                print('>> S8 already computed')  

        if label == '1x2pt' or label == '2x2pt' or label == '3x2pt':
            color='purple'
            symbol = '*'
            size = 13.0
        elif 'HSC Year 1' in label:
            color = 'red'
            size = 8.0
        elif ('DES' in label) or ('HSC Year 3' in label) or ('KiDS' in label):
            color='blue'
            size = 8.0
        elif 'Planck' in label:
            color='green'
            size = 8.0
        else:
            color='k'
            symbol = 'o'
            size = 8.0
        # y-axis position where to put this case
        ypos = row_index[index] 
        ##########
        ### S8 ###
        ##########
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$S_8$')], weights = weights)
        # med,up,lo=report_mean_68assy(arr=sample[:,parameters.index('$S_8$')])
        print(f'S8 = {med}+{up}-{lo}')
        # Plot data with asymetric errorbars in x-axis direction
        ax1.errorbar(med, ypos, xerr=[[lo],[up]], fmt=symbol, ms = size, color=color, capsize=0, capthick=1,elinewidth=3)
        if label == '1x2pt' or label == 'Planck 2018 TTTEEE':
            # Set vertical shaded region spanning the 1-sigma confidence interval
            ax1.axvspan(med-lo, med+up, alpha=0.2, color=color)
            # Set a vertical line at x = central value of S8
            ax1.axvline(med, lw=0.5, color=color)    
        ###############
        ### Omega_m ###
        ###############
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\Omega_m$')], weights = weights)
        # med,up,lo=report_mean_68assy(arr=sample[:,parameters.index('$\Omega_m$')])
        print(f'Omega_matter = {med}+{up}-{lo}')
        # Plot data with asymetric errorbars in x-axis direction
        ax2.errorbar(med, ypos, xerr=[[lo],[up]], fmt=symbol, ms = size, color=color, capsize=0, capthick=1, elinewidth=3)
        if label == '1x2pt' or label == 'Planck 2018 TTTEEE':
            # Set vertical shaded region spanning the 1-sigma confidence interval
            ax2.axvspan(med-lo, med+up, alpha=0.2, color=color)
            # Set a vertical white line at x = central value of S8
            ax2.axvline(med, lw=0.5, color=color)    
        ##############
        ### sigma8 ###
        ##############
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\sigma_8$')], weights = weights)
        # med,up,lo=report_mean_68assy(arr=sample[:,parameters.index('$\sigma_8$')])
        print(f'sigma_8 = {med}+{up}-{lo}')
        # Plot data with asymetric errorbars in x-axis direction
        ax3.errorbar(med, ypos, xerr=[[lo],[up]], fmt=symbol, ms = size, color=color, capsize=0, capthick=1, elinewidth=3)
        if label == '1x2pt' or label == 'Planck 2018 TTTEEE':
            # Set vertical shaded region spanning the 1-sigma confidence interval
            ax3.axvspan(med-lo, med+up, alpha=0.2, color=color)
            # Set a vertical white line at x = central value of S8
            ax3.axvline(med, lw=0.5, color=color)    
        # Plot a grey translucent horizontal line at the y - 0.5 position
        ax1.axhline(ypos - 0.5, color='grey', lw=2.0, alpha=0.3)
        ax2.axhline(ypos - 0.5, color='grey', lw=2.0, alpha=0.3)
        ax3.axhline(ypos - 0.5, color='grey', lw=2.0, alpha=0.3)

        if label == '1x2pt':
            # Set text in top left of the first panel stating "This work"
            ax1.text(0.47, ypos, 'This work', fontsize=20, alpha=0.6)
        elif label == '3x2pt - NLA with $\\eta_{\\text{eff}}$=0':
            ax1.text(0.47, ypos, 'Robustness tests', fontsize=20, alpha=0.6)
        elif label == '3x2pt - $\\sum m_\\nu$ = 0.06 eV':
            ax1.text(0.47, ypos, 'Extensions', fontsize=20, alpha=0.6)
        elif label == 'HSC Year 1 (Hikage et al.) - 1x2pt C$_\ell$':
            ax1.text(0.47, ypos, 'Firecrown validation', fontsize=20, alpha=0.6)
        elif label == 'DES Year 3 $\\xi$(r) - 1x2pt':
            ax1.text(0.47, ypos, 'Stage III & CMB', fontsize=20, alpha=0.6)

        if label in ['3x2pt', 'Shear + Clustering', '1x2pt wCDM','HSC Year 1 (re-analysis) - 1x2pt C$_\ell$']:
            ax1.axhline(ypos - 0.5, color='grey', alpha=0.5, lw=4.0)
            ax2.axhline(ypos - 0.5, color='grey', alpha=0.5, lw=4.0)
            ax3.axhline(ypos - 0.5, color='grey', alpha=0.5, lw=4.0)

        # if fname == '/pscratch/sd/d/davidsan/Planck2018_chains/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE.txt':
        #     ax3.axvline(med-lo, ls='--', lw=0.5, color='k')
        #     ax3.axvline(med+up, ls='--', lw=0.5, color='k')
        # Append the color to the color array
        color_array.append(color)
        # Next loop
        index += 1    

    # Set as y-tick the name of the chain
    ax1.set_yticks(row_index, label_list, fontsize=20)
    # Get the current y-tick labels
    yticks = ax1.get_yticklabels()
    # Apply bold style to specific y-tick labels
    for i in bold_indices:
        yticks[i].set_fontweight('bold')

    # Change color and apply bold style to specific y-tick labels
    for i in np.arange(len(color_array)):
        yticks[i].set_color(color_array[i])
    # Set the x-axis limits
    ax1.set_xlim([0.45,0.95])
    ax2.set_xlim([0.1,0.5])
    ax3.set_xlim([0.5,1.3])

    savepath = '/pscratch/sd/d/davidsan/HSC-PDR1-3x2pt-harmonic-methods/figures/S8-Omegam-sigma8-comparison/'
    if labelpng:
        plt.savefig(os.path.join(savepath, f'S8-median1D-{labelpng}.png'),
                bbox_inches='tight',
                dpi=300)

        plt.savefig(os.path.join(savepath, f'S8-median1D-{labelpng}.pdf'),
                bbox_inches='tight',
                dpi=300)

    plt.show()
    plt.close()# 