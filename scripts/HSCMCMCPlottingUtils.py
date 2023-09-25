# Use desc-python-bleed to avoid Latex issues
import os
import sys
sys.path.insert(0, '/global/homes/d/davidsan/ChainConsumer')
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('text', usetex=True)
# plt.style.use('/global/cscratch1/sd/davidsan/dsc_custom.mplstyle')
import scipy.stats as stats
from numpy.random import normal, uniform

# Matplotlib formatting
# Matplotlib settings
colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

plt.rcParams['figure.figsize'] = 3., 3.
plt.rcParams['figure.dpi'] = 300
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
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.pad'] = 6.
plt.rcParams['xtick.minor.pad'] = 6.
plt.rcParams['ytick.major.pad'] = 6.
plt.rcParams['ytick.minor.pad'] = 6.
plt.rcParams['xtick.major.size'] = 6. # major tick size in points
plt.rcParams['xtick.minor.size'] = 3. # minor tick size in points
plt.rcParams['ytick.major.size'] = 6. # major tick size in points
plt.rcParams['ytick.minor.size'] = 3. # minor tick size in points
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] =  'serif'
# plt.rcParams['font.family'] =  'cmr10'
plt.rcParams['font.size'] = 8 
# axes linewidth
plt.rcParams['axes.linewidth'] = 1.2 #set the value globally

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
                        'cosmological_parameters--h0':'$h$',
                        'cosmological_parameters--a_s':'$A_s$',
                        'cosmological_parameters--log10as':'$\ln(10^{10} A_s)$',
                        'cosmological_parameters--log10as_hamana':'$\ln(10^{9} A_s)$',
                        'cosmological_parameters--n_s':'$n_{s}$',
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
                        'COSMOLOGICAL_PARAMETERS--SIGMA_8':'$\sigma_8$',
                        'COSMOLOGICAL_PARAMETERS--SIGMA_12':'$\sigma_12$',
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
               # Lens photo-z uncert.
               '$\Delta z^{lens}_1$':[-0.08,0.08],
               '$\Delta z^{lens}_2$':[-0.08,0.08],
               '$\Delta z^{lens}_3$':[-0.08,0.08],
               '$\Delta z^{lens}_4$':[-0.08,0.08],
               '$\sigma z^{lens}_1$':[0.9,1.1],
               '$\sigma z^{lens}_2$':[0.9,1.1],
               '$\sigma z^{lens}_3$':[0.9,1.1],
               '$\sigma z^{lens}_4$':[0.9,1.1],
               # Lens galaxy bias 
               '$b^{lens}_1$':[1.0,2.0],
               '$b^{lens}_2$':[1.0,2.0],
               '$b^{lens}_3$':[1.0,2.0],
               '$b^{lens}_4$':[1.0,2.0],
               # Sources photo-z uncert.
               # '$\Delta z^{source}_1$':[-0.008,0.008],
               # '$\Delta z^{source}_2$':[-0.008,0.008],
               # '$\Delta z^{source}_3$':[-0.008,0.008],
               # '$\Delta z^{source}_4$':[-0.008,0.008],
               '$\Delta z^{source}_1$':[-0.1,0.1],
               '$\Delta z^{source}_2$':[-0.1,0.1],
               '$\Delta z^{source}_3$':[-0.1,0.1],
               '$\Delta z^{source}_4$':[-0.1,0.1],
               '$\sigma z^{source}_1$':[0.9,1.1],
               '$\sigma z^{source}_2$':[0.9,1.1],
               '$\sigma z^{source}_3$':[0.9,1.1],
               '$\sigma z^{source}_4$':[0.9,1.1],
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

########################################
###         Initialize chains        ###
########################################
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
    # print(header)
    # Initialize empty list
    parameters_latex = list()
    # Obtain corresponding latex labels
    if 'hikage' in fname:
        for par in parameters_cosmosis:
            if par in parameters_dict_desc.keys():
                # print(parameters_dict_desc[par])
                parameters_latex.append(parameters_dict_desc[par])
    elif 'hamana' in fname:
        for par in parameters_cosmosis:
            if par in parameters_dict_desc_hamana.keys():
                parameters_latex.append(parameters_dict_desc_hamana[par])
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
    # plt.hist(om, histtype='step')
    # plt.title('$\Omega_m$')
    # plt.show()
    # plt.close()
    # print(parameters.index('$\\Omega_m$'))
    # print('om',np.mean(om))
    sigma8=sample[:,parameters.index('$\\sigma_8$')]
    # plt.hist(sigma8, histtype='step')
    # plt.title('$\sigma_8$')
    # plt.show()
    # plt.close()
    # print(parameters.index('$\\sigma_8$'))
    # print('sigma8',np.mean(sigma8))
    # Compute S8
    S8=sigma8*(om/0.3)**alpha
    # plt.hist(S8, histtype='step')
    # plt.title('$S_8$')
    # plt.show()
    # plt.close()
    # print('S8',np.mean(S8))
    # Add column to the main sample
    sample = np.c_[sample,S8]
    parameters.append('$S_8$')
    
    return(sample,parameters)

def generate_cosmosis_chain(fname_list,chain_label_list,add_hsc_hikage=False,add_hsc_hamana=False,add_nicola=False,add_prior=True):
    # fname - path to txt chain (could be a list of chains to compare)
    # add_hsc add comparison with HSC contours
    
    # Output - chainconsumer chain
    fname_hsc = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/hsc_chains/using_cls/HSC_Y1_LCDM_post_fid.txt')
    fname_hsc_hamana = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/hsc_chains/using_corr_function/HSC_hamana2020_fiducial/hsc_hamana2020_fiducial.txt')
    fname_nicola = os.path.join('/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/carlos_chains/hsc_dz_Andrina_ell300/hsc_dz_Andrina_ell300.merged.txt')
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
            sample = np.loadtxt(fname)
            weigths = sample[:,0]
            plt.plot(np.arange(len(weigths)), weigths)
            plt.show()
            plt.close()
            posterior = sample[:,1]
            parameters = list(np.copy(parameters_hsc))
            # Appendin S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=0.5)
            fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))
            
            axs[0].hist(sample[:,parameters.index('$\Omega_{cdm} \cdot h^2$')],histtype='step')
            axs[0].set_title('$\Omega_{cdm} \cdot h^2$')
            
            axs[1].hist(sample[:,parameters.index('$\Omega_b \cdot h^2$')],histtype='step')
            axs[1].set_title('$\Omega_b \cdot h^2$')
                           
            axs[2].hist(sample[:,parameters.index('$\Omega_m$')],histtype='step')
            axs[2].set_title('$\Omega_m$')
            
            axs[3].hist(sample[:,parameters.index('$\sigma_8$')],histtype='step')
            axs[3].set_title('$\sigma_8$')
            
            axs[4].hist(sample[:,parameters.index('$S_8$')],histtype='step')
            axs[4].set_title('$S_8$')

            axs[5].hist(sample[:,parameters.index('$\ln(10^{10} A_s)$')],histtype='step')
            axs[5].set_title('$\ln(10^{10} A_s)$')
            
            plt.show()
            plt.close()
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
            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weigths,
                        # posterior=posterior,
                        kde=kde,
                        name=chain_label)
        elif fname == fname_nicola:
            sample = np.loadtxt(fname)
            weigths = sample[:,0]
            posterior = sample[:,1]
            parameters = list(np.copy(parameters_nicola))
            # Sampling Om_c and Om_b
            sample,parameters=omega_m(sample=sample,parameters=parameters)
            # Appendin S8 derived parameter
            sample,parameters = S8(sample=sample,parameters=parameters,alpha=0.5)
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18, 3))
            
            axs[0].hist(sample[:,parameters.index('$\Omega_{cdm}$')],histtype='step')
            axs[0].set_title('$\Omega_{cdm}$')
            
            axs[1].hist(sample[:,parameters.index('$\Omega_b$')],histtype='step')
            axs[1].set_title('$\Omega_b$')
                           
            axs[2].hist(sample[:,parameters.index('$\Omega_m$')],histtype='step')
            axs[2].set_title('$\Omega_m$')
            
            axs[3].hist(sample[:,parameters.index('$\sigma_8$')],histtype='step')
            axs[3].set_title('$\sigma_8$')
            
            axs[4].hist(sample[:,parameters.index('$S_8$')],histtype='step')
            axs[4].set_title('$S_8$')
            
            plt.show()
            plt.close()
            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weigths,
                        # posterior=posterior,
                        kde=kde,
                        name=chain_label)
            
        elif fname == fname_hsc_hamana:
            sample = np.loadtxt(fname)
            weights = sample[:,0]
            # Weights checkin
            plt.plot(np.arange(len(weights)),weights)
            plt.show()
            plt.close()
            # posterior = sample[:,1]
            parameters = list(np.copy(parameters_hsc_hamana))
            # print(parameters)
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
            # Histograms
            fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))
            
            axs[0].hist(sample[:,parameters.index('$\Omega_{cdm}$')],histtype='step')
            axs[0].set_title('$\Omega_{cdm}$')
            
            axs[1].hist(sample[:,parameters.index('$\Omega_b$')],histtype='step')
            axs[1].set_title('$\Omega_b$')
                           
            axs[2].hist(sample[:,parameters.index('$\Omega_m$')],histtype='step')
            axs[2].set_title('$\Omega_m$')
            
            axs[3].hist(sample[:,parameters.index('$\sigma_8$')],histtype='step')
            axs[3].set_title('$\sigma_8$')
            
            axs[4].hist(sample[:,parameters.index('$S_8$')],histtype='step')
            axs[4].set_title('$S_8$')

            axs[5].hist(sample[:,parameters.index('$\ln(10^{9} A_s)$')],histtype='step')
            axs[5].set_title('$\ln(10^{9} A_s)$')
            
            plt.show()
            plt.close()
            
            # fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
            # 
            # axs[0].hist(sample[:,parameters.index('$\Delta z^{source}_1$')],histtype='step')
            # axs[0].set_title('$\Delta z^{source}_1$')
            # 
            # axs[1].hist(sample[:,parameters.index('$\Delta z^{source}_2$')],histtype='step')
            # axs[1].set_title('$\Delta z^{source}_2$')
            #                
            # axs[2].hist(sample[:,parameters.index('$\Delta z^{source}_3$')],histtype='step')
            # axs[2].set_title('$\Delta z^{source}_3$')
            # 
            # axs[3].hist(sample[:,parameters.index('$\Delta z^{source}_4$')],histtype='step')
            # axs[3].set_title('$\Delta z^{source}_4$')
            # 
            # plt.show()
            # plt.close()
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
                weigths = np.ones(sample.shape[0])
                posterior = sample[:,-1]
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
                    
                # Sampling Om_c*h^2 and Om_b*h^2
                sample,parameters=omega_m(sample=sample,parameters=parameters)
                # Computing S_8
                sample,parameters=S8(sample=sample,parameters=parameters,alpha=0.5)
                # print(sample,parameters)
            else:
                print('>> Chain with weights')
                # Our MCMC
                sample = np.loadtxt(fname)
                # Extract parameters name from chain
                parameters = cosmosis_header(fname=fname)
                # Extract weights and posterior
                weigths = sample[:,parameters.index('$weight$')]
                plt.plot(np.arange(len(weigths)),weigths)
                plt.show()
                plt.close()
                posterior = sample[:,parameters.index('$post$')]
                # Sampling Om_c*h^2 and Om_b*h^2
                sample,parameters=omega_m(sample=sample,parameters=parameters)
                # Computing S_8
                sample,parameters=S8(sample=sample,parameters=parameters,alpha=0.5)

            fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))

            if '$\Omega_{cdm} \cdot h^2$' in parameters:
                axs[0].hist(sample[:,parameters.index('$\Omega_{cdm} \cdot h^2$')],histtype='step')
                axs[0].set_title('$\Omega_{cdm} \cdot h^2$')
            elif '$\Omega_{cdm}$' in parameters:
                axs[0].hist(sample[:,parameters.index('$\Omega_{cdm}$')],histtype='step')
                axs[0].set_title('$\Omega_{cdm}$')

            if '$\Omega_b \cdot h^2$' in parameters:
                axs[1].hist(sample[:,parameters.index('$\Omega_b \cdot h^2$')],histtype='step')
                axs[1].set_title('$\Omega_b \cdot h^2$')
            elif '$\Omega_b$' in parameters:
                axs[1].hist(sample[:,parameters.index('$\Omega_b$')],histtype='step')
                axs[1].set_title('$\Omega_b$')

            axs[2].hist(sample[:,parameters.index('$\Omega_m$')],histtype='step')
            axs[2].set_title('$\Omega_m$')

            axs[3].hist(sample[:,parameters.index('$\sigma_8$')],histtype='step')
            axs[3].set_title('$\sigma_8$')

            axs[4].hist(sample[:,parameters.index('$S_8$')],histtype='step')
            axs[4].set_title('$S_8$')

            if '$\ln(10^{10} A_s)$' in parameters:
                axs[5].hist(sample[:,parameters.index('$\ln(10^{10} A_s)$')],histtype='step')
                axs[5].set_title('$\ln(10^{10} A_s)$')
            elif '$\ln(10^{9} A_s)$' in parameters:
                axs[5].hist(sample[:,parameters.index('$\ln(10^{9} A_s)$')],histtype='step')
                axs[5].set_title('$\ln(10^{9} A_s)$')

            plt.show()
            plt.close()

            # Add to already initialize chain
            c.add_chain(sample,
                        parameters=parameters,
                        weights=weigths,
                        posterior=posterior,
                        kde=kde,
                        name=chain_label)
            
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\Omega_m$')])
        print(f'Omega_matter = {med}+{up}-{lo}')
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$\sigma_8$')])
        print(f'sigma_8 = {med}+{up}-{lo}')
        med,up,lo=report_median_68assy(arr=sample[:,parameters.index('$S_8$')])
        print(f'S8 = {med}+{up}-{lo}')
        # if report_median == True:
        # for par in parameters:
        #     med,up,lo=report_median_68assy(arr=sample[:,parameters.index(par)])
        #     print(f'{par} = {med}+{up}-{lo}')
        print("-------------------------------")
        k += 1
    # Config chain
    c.configure(flip=False,
                linestyles=["-"]*len(fname_list),
                # linewidths=[1.0]+[1.2]*len(fname_list),
                shade=[False]+[False]*len(fname_list),
                legend_kwargs={"fontsize": 5},#, "loc": "upper right"},
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

def report_median_68assy(arr):
    # arr = data_hsc[:,parameters_hsc.index('$\Omega_m$')]
    median = np.median(arr)
    upper, lower = sigma_68(arr, axis=None)
    upper = upper - median
    lower = median - lower 
    median = np.round(median, 3)
    upper = np.round(upper, 3)
    lower = np.round(lower, 3)
    # print(f'{median}+{upper}-{lower}')
    return(median, upper, lower)

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
def plot_Omegam_sigma8_S8(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$S_8$'], 
                             extents=extents_dict,
                             watermark=r"Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'Om_sigma8_S8_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_Omegam_sigma8_lnAs(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$', '$\sigma_8$', '$\ln(10^{10} A_s)$'], 
                             extents=extents_dict,
                             watermark=r"Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'Om_sigma8_lnAs_{labelpng}.png'),
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
                             extents=extents_dict,
                             watermark="Preliminary")
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
                             watermark="Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'Om_S8_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\sigma_8$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_hamana(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\sigma_8$','$\ln(10^{9} A_s)$', '$\Omega_b$','$\Omega_{cdm}$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'cosmo_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_as(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_m$','$\ln(10^{10} A_s)$','$\Omega_b \cdot h^2$','$\Omega_{cdm} \cdot h^2$','$n_{s}$','$h$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize=(5,5))
    plt.savefig(os.path.join(savepath,f'cosmo_as_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_cosmological_def(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\Omega_{cdm} \cdot h^2$','$\Omega_b \cdot h^2$','$h$','$n_{s}$','$\ln(10^{10} A_s)$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize=(5,5))
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
                             watermark="Preliminary",
                             figsize="column")
    plt.savefig(os.path.join(savepath,f'delta_pz_lens_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_pz_stretch_lens(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\sigma z^{lens}_1$','$\sigma z^{lens}_2$','$\sigma z^{lens}_3$','$\sigma z^{lens}_4$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize="column")
    plt.savefig(os.path.join(savepath,f'stretch_pz_lens_{labelpng}.png'),
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
                             watermark="Preliminary",
                             figsize="column")
    plt.savefig(os.path.join(savepath,f'delta_pz_source_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_pz_stretch_source(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$\sigma z^{source}_1$','$\sigma z^{source}_2$','$\sigma z^{source}_3$','$\sigma z^{source}_4$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize="column")
    plt.savefig(os.path.join(savepath,f'stretch_pz_source_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###          Galaxy bias           ###
######################################
def plot_galbias_lens(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
    fig = chain.plotter.plot(parameters=['$b^{lens}_1$','$b^{lens}_2$','$b^{lens}_3$','$b^{lens}_4$'],
                             extents=extents_dict,
                             watermark="Preliminary",
                             figsize="column")
    plt.savefig(os.path.join(savepath,f'delta_pz_lens_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

def plot_intrinsic_alignments(chain,labelpng,mode="lin",savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/shear'):
    # mode = "lin" or "nla" for linear o non-linear alignment, respectively
    if mode == "nla":
        fig = chain.plotter.plot(parameters=['$A_{IA}$','$\eta$'],
                                 extents=extents_dict,
                                 watermark="Preliminary",
                                 figsize=(5,5))
        plt.savefig(os.path.join(savepath,f'nla-ia_{labelpng}.png'),
                   dpi=300,
                   bbox_inches='tight')
    elif mode == "lin":
        fig = chain.plotter.plot(parameters=['$A_{IA,lin}$','$\\alpha_z$'],
                                 extents=extents_dict,
                                 watermark="Preliminary",
                                 figsize=(5,5))
        plt.savefig(os.path.join(savepath,f'lin-ia_{labelpng}.png'),
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
                             watermark="Preliminary",
                             figsize=(3,3))
    plt.savefig(os.path.join(savepath,f'multiplicative_shear_bias_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)

######################################
###      Full-triangle plot        ###
######################################

def plot_full_shear_hikage(chain,labelpng,savepath='/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/chains/figures/clustering'):
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
                             watermark="Preliminary",
                             figsize="GROW")
    plt.savefig(os.path.join(savepath,f'full_shear_hikage_{labelpng}.png'),
               dpi=300,
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
                             watermark="Preliminary",
                             figsize="GROW")
    plt.savefig(os.path.join(savepath,f'full_shear_hamana_{labelpng}.png'),
               dpi=300,
               bbox_inches='tight')
    plt.show()
    plt.close()
    return(fig)
