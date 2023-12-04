import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from scipy.special import erf
# %matplotlib inline
import os 
import sacc
import datetime
import time
sys.path.append('/pscratch/sd/d/davidsan/HSC-PDR1-3x2pt-harmonic-methods/scripts/')
import HSCMeasurementUtils as hmu

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

########################################
###   Galaxy clustering scale cuts   ###
########################################
# Parametrization used in Nicola et al. https://arxiv.org/abs/1912.08209
class HaloProfileHOD(ccl.halos.HaloProfileNFW):
    def __init__(self, c_M_relation,
                 lMmin=12.02, lMminp=-1.34,
                 lM0=6.6, lM0p=-1.43,
                 lM1=13.27, lM1p=-0.323):
        self.lMmin=lMmin
        self.lMminp=lMminp
        self.lM0=lM0
        self.lM0p=lM0p
        self.lM1=lM1
        self.lM1p=lM1p
        # Pivot is z_p = 0.65
        self.a0 = 1./(1+0.65)
        self.sigmaLogM = 0.4
        self.alpha = 1.
        super(HaloProfileHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod

    def _lMmin(self, a):
        return self.lMmin + self.lMminp * (a - self.a0)

    def _lM0(self, a):
        return self.lM0 + self.lM0p * (a - self.a0)

    def _lM1(self, a):
        return self.lM1 + self.lM1p * (a - self.a0)

    def _Nc(self, M, a):
        # Number of centrals
        Mmin = 10.**self._lMmin(a)
        return 0.5 * (1 + erf(np.log(M / Mmin) / self.sigmaLogM))

    def _Ns(self, M, a):
        # Number of satellites
        M0 = 10.**self._lM0(a)
        M1 = 10.**self._lM1(a)
        return np.heaviside(M-M0,1) * ((M - M0) / M1)**self.alpha

    def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Nc[:, None] * (1 + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Ns[:, None] * uk
        prof = Nc[:, None] * (2 * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
class Profile2ptHOD(ccl.halos.Profile2pt):
    def fourier_2pt(self, prof, cosmo, k, M, a,
                      prof2=None, mass_def=None):
        return prof._fourier_variance(cosmo, k, M ,a, mass_def)
    
def GenerateHODClustering(cosmo, pk_ggf, apply_scalecuts=False):
    # Initialize empty sacc
    S = sacc.Sacc()
    # Metadata
    nbin_lens = 4
    S.metadata['nbin_lens'] = nbin_lens
    S.metadata['creator'] = 'David'
    S.metadata['creation'] = datetime.datetime.now().isoformat()
    S.metadata['info'] = 'HOD Galaxy clustering - Theory predictions using CCL'
    ############################
    ###   Lens sample dndz   ###
    ############################
    # Reading our lens sample dndz
    fname = '/pscratch/sd/d/davidsan/HSC-PDR1-3x2pt-harmonic-methods/data/harmonic/txpipe/source_s16a_lens_dr1/all-fields/dndz/summary_statistics_fourier_all_SourcesS16A_LensesDR1_pz_mc_eab_HikageShearSC.sacc'
    s = sacc.Sacc.load_fits(fname)
    for i in np.arange(nbin_lens):
        # Add the appropriate tracer
        print(f"Adding lens dndz z-bin {i+1} ...")
        z = s.tracers[f'lens_{i}'].z
        nz = s.tracers[f'lens_{i}'].nz
        S.add_tracer('NZ', f'lens_{i}', z, nz)
        plt.plot(z, nz)
    plt.show()
    plt.close()
    #########################
    ###   Mock HOD data   ###
    #########################
    # HSC 3x2pt re-analysis project multipoles (nside = 2048)
    ell, Cell = s.get_ell_cl("galaxy_density_cl", "lens_0", "lens_0", return_cov=False)
    # l_arr = np.array([149.5,249.5,349.5,499.5,699.5,899.5,1199.5,1599.5,1999.5])
    l_arr = np.copy(ell)
    print('>> multipoles = ', l_arr)
    # Initialize data type
    galaxy_density_cl = sacc.standard_types.galaxy_density_cl
    # Considering auto- and cross- correlations
    for i in np.arange(nbin_lens):
        for j in np.arange(nbin_lens):
            if i >= j:
                # print(i,j)
                # print(i)
                nz_arr_1 = s.tracers[f'lens_{i}'].nz
                nz_arr_2 = s.tracers[f'lens_{j}'].nz 

                z_arr_1 = s.tracers[f'lens_{i}'].z
                z_arr_2 = s.tracers[f'lens_{j}'].z

                if np.all(z_arr_1 == z_arr_2):
                    z_arr = z_arr_1

                # Generating galaxy tracers
                t_g_1 = ccl.NumberCountsTracer(cosmo=cosmo,
                                               has_rsd=False, 
                                               dndz=(z_arr, nz_arr_1),
                                               bias=(z_arr, np.ones_like(z_arr)))

                t_g_2 = ccl.NumberCountsTracer(cosmo=cosmo,
                                               has_rsd=False,
                                               dndz=(z_arr, nz_arr_2),
                                               bias=(z_arr, np.ones_like(z_arr)))

                # Computing angular power spectrum
                cl_gg = ccl.angular_cl(cosmo, t_g_1, t_g_2, l_arr, p_of_k_a=pk_ggf)

                bin_name_1 = f'lens_{i}'
                bin_name_2 = f'lens_{j}'

                # Add the values
                S.add_ell_cl(galaxy_density_cl, bin_name_1, bin_name_2, l_arr, cl_gg)
                
    ######################
    ###   Covariance   ###
    ######################
    print('>> Covariance matrix shape BEFORE removing GGL and Shear')
    print(s.covariance.covmat.shape)
    # Cut dv to just have galaxy clustering part ...
    s.remove_selection(data_type='galaxy_shearDensity_cl_e')
    s.remove_selection(data_type='galaxy_shear_cl_ee')
    print('>> Covariance matrix shape AFTER removing GGL and Shear')
    print(s.covariance.covmat.shape)
    print('>> Length of the signal: ', len(s.mean))
    # Extract galaxy clustering covariance matrix ...
    cov = s.covariance.covmat
    # Introduce the covariance matrix into our new HOD galaxy clustering theoretical prediction data vector...
    print('>> Introducing covariance matrix')
    S.add_covariance(cov)
    # Plotting the correlation matrix
    Covariance_Plot(s = S, savefig=False)
    ################################
    ###   Save the data vector   ###
    ################################
    path_save = '/pscratch/sd/d/davidsan/HSC-PDR1-3x2pt-harmonic-methods/data/harmonic/hod_clustering'
    S.save_fits(os.path.join(path_save,'summary_statistics_clustering_hod_rsd.fits'), overwrite=True)
    # Maxmium Physical scales to generate data vectors
    # with scale cuts
    kmax_array = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5])
    # Compute effective redshift of each bin and its
    # corresponding comoving distance assuming Planck cosmology
    zeff_dict, chi_dict = zEffective_Comoving_Dist_Calculation(s = S,
                                                               cosmo = cosmo)
    if apply_scalecuts == True:
        print('>> Producing DATA VECTORS with scale cuts')
        # For a given k_max translate to maximum multipole
        # and apply the scale cut
        for kmax in kmax_array[::-1]:
            print('#######################')
            print(f'k_max = {kmax} [1/Mpc]')
            print('#######################')
            ellmax_dict = Kmax_to_Ellmax(kmax = kmax,
                                         chi_dict = chi_dict)
            print('Number of remaining data points:')
            for i in np.arange(nbin_lens):
                for j in np.arange(nbin_lens):
                    if i >= j:
                        # Extract ell_max
                        cut = ellmax_dict[f'{i}_{j}']
                        S.remove_selection(data_type='galaxy_density_cl', tracers=[f'lens_{i}', f'lens_{j}'], ell__gt=cut)
                        ell, Cell = S.get_ell_cl("galaxy_density_cl", f'lens_{i}', f'lens_{j}', return_cov=False)
                        print(f'Corr {i}{j} = {len(Cell)}')
            # Save data vector
            print('>> Saving data vector ...')
            S.save_fits(os.path.join(path_save,f'summary_statistics_clustering_hod_rsd_kmax_{str(np.round(kmax, 2))}.sacc'), overwrite=True)
    return(S)

def zEffective_Comoving_Dist_Calculation(s, cosmo):
    # Compute the effective redshift per tomographic bin
    zeff_dict = {}
    chi_dict = {}
    # N(z) -- Lenses & Sources
    nbin_lens = 4
    for i in np.arange(nbin_lens):
        for j in np.arange(nbin_lens):
            if i >= j:
                print('Z-BIN COMBINATION',i, j)
                # Normalizations
                norm_1 = np.sum(s.tracers[f'lens_{i}'].nz)
                norm_2 = np.sum(s.tracers[f'lens_{j}'].nz)

                dist_1 = s.tracers[f'lens_{i}'].nz/norm_1
                dist_2 = s.tracers[f'lens_{j}'].nz/norm_2

                # Sum of distributions and normalize
                dist_total = dist_1 + dist_2
                dist_total = dist_total / np.sum(dist_total)

                if i != j:
                    mean = np.sum(s.tracers[f'lens_{i}'].z*dist_total)
                    # print(mean)
                    zeff_dict[f'{i}_{j}'] = mean
                else:
                    mean_1 = np.sum(s.tracers[f'lens_{i}'].z*dist_1) # Computed as the mean redshift ref. A. Nicola et al. 2020
                    mean_2 = np.sum(s.tracers[f'lens_{j}'].z*dist_2)
                    # print(mean_1, mean_2)
                    mean = np.copy(mean_1)
                zeff_dict[f'{i}_{j}'] = mean
                print(f'z-bin = {i, j}, z_eff = {np.round(mean, 3)}')
                # Computing comoving distance
                chi = ccl.background.comoving_radial_distance(cosmo, a=1./(1+mean))
                chi_dict[f'{i}_{j}'] = chi
                print(f'Comoving distance = {np.round(chi,2)} Mpc')
    return(zeff_dict, chi_dict)

def Kmax_to_Ellmax(kmax, chi_dict):
    # Initialize maximum multipole dictionary
    ellmax_dict = {}
    nbin_lens = 4
    for i in np.arange(nbin_lens):
        for j in np.arange(nbin_lens):
            if i >= j:
                print('Correlation = ', i , j)
                chi_max = chi_dict[f'{i}_{j}']
                # transform from kmax to ellmax given the comoving distance
                ellmax_dict[f'{i}_{j}'] = kmax * chi_max
                print(f'ell_max = {int(kmax * chi_max)}')
    return(ellmax_dict)

