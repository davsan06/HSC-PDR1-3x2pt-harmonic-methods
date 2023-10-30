from wsgiref.util import shift_path_info
import sacc
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import healpy as hp
import seaborn as sns

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
plt.rcParams['xtick.major.size'] = 6. # major tick size in points
plt.rcParams['xtick.minor.size'] = 3. # minor tick size in points
plt.rcParams['ytick.major.size'] = 6. # major tick size in points
plt.rcParams['ytick.minor.size'] = 3. # minor tick size in points
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] =  'serif'
# plt.rcParams['font.serif'] = 'Computer Modern Roman Bold'
plt.rcParams['font.size'] = 18  

colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", '#800080', "#0072B2", "#CC79A7", "#D55E00"]

###################################
###  TXPipe consistency checks  ###
###################################

def IndexToDeclRa(index,nside):
    theta,phi=hp.pixelfunc.pix2ang(nside,index)
    return -np.degrees(theta-np.pi/2.),np.degrees(np.pi*2.-phi)

def RaDecToIndex(ra_deg, dec_deg, nside=32):
    """
    Convert Right Ascension (RA) and Declination (Dec) coordinates to Healpix pixel indices.

    Parameters:
    - ra_deg: Numpy array of Right Ascension values in degrees.
    - dec_deg: Numpy array of Declination values in degrees.
    - nside: Healpix resolution parameter (default is 32).

    Returns:
    - Numpy array of Healpix pixel indices corresponding to the input coordinates.
    """
    # Convert RA and Dec to pixel indices
    pix_indices = hp.ang2pix(nside, np.radians(90 - dec_deg), np.radians(ra_deg))

    return pix_indices

def LensTomoCat_plot(fname,title):
    # Read table
    table = h5py.File(fname)
    # Extract info
    zbin = np.array(table['tomography']['bin'])
    counts = np.array(table['tomography']['counts_2d'][0])
    binning = np.unique(zbin)
    # Plot histogram
    plt.hist(zbin, 
             bins=binning,
             align='left',
             histtype='step',
             color='k',
             label=f'total = {counts}')
    plt.title('Lens counts')
    plt.xlabel('redshift bin')
    plt.ylabel('counts')
    plt.legend(frameon=False)
    return()

def LensMaps_plot(fname,title):
    table = h5py.File(fname)

    fig, axes = plt.subplots(1,4,figsize=(18,3))
    fig_w, axes_w = plt.subplots(1,4,figsize=(18,3))

    for i in np.arange(4):
        # Number of galaxies
        ngal = table['maps'][f'ngal_{i}']['value']
        pixel = table['maps'][f'ngal_{i}']['pixel']
        theta,phi = IndexToDeclRa(pixel,nside)
        sc_ngal = axes[i].scatter(phi,theta,c=ngal,s=1.0)
        axes[i].set_xlabel('R.A.')
        if i == 0:
            axes[i].text(0.7,0.8,f'{title}',transform = axes[i].transAxes)
            axes[i].set_ylabel('Dec.')

        # Weighted number of galaxies
        ngal_w = table['maps'][f'weighted_ngal_{i}']['value']
        pixel = table['maps'][f'weighted_ngal_{i}']['pixel']
        theta,phi = IndexToDeclRa(pixel,nside)
        sc_ngal_w = axes_w[i].scatter(phi,theta,c=ngal_w,s=1.0)
        axes_w[1].set_xlabel('R.A.')
        if i == 0:
            axes_w[i].text(0.7,0.8,f'{title}',transform = axes_w[i].transAxes)
            axes_w[i].set_ylabel('Dec.')
    cbar = fig.colorbar(sc_ngal)
    cbar.set_label('lens counts')
    cbar_w = fig_w.colorbar(sc_ngal_w)
    cbar_w.set_label('weighted lens counts')
    return()

def Mask_plot(fname,title):
    table = h5py.File(fname)
    mask = table['maps']['mask']['value']
    pixel = table['maps']['mask']['pixel']
    theta,phi = IndexToDeclRa(pixel,nside)
    sc_mask = plt.scatter(phi,theta,c=mask,s=1.0)
    plt.title(f'{title} (Mask)')
    plt.xlabel('R.A.')
    plt.ylabel('Dec.')
    cbar = plt.colorbar(sc_mask)
    cbar.set_label('fracdet')
    return()

def RedshiftDistr_plot(sacc, savepath = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3*1.5, 4*1.5), sharex=True)
    plt.subplots_adjust(hspace=0)

    nbins_src = sum(1 for elemento in sacc.tracers.keys() if "source" in elemento)
    nbins_lens = sum(1 for elemento in sacc.tracers.keys() if "lens" in elemento)

    # Sources
    z = sacc.tracers['source_0'].z
    delta_z = z[1] - z[0]
    for i in np.arange(nbins_src):
        nz = sacc.tracers[f'source_{i}'].nz
        area = sum(delta_z * nz)
        # area normalization
        nz /= area
        if i == 0:
            ax1.plot(z,nz,color=colors[i], label='Sources')
            ax1.fill_between(z,nz, 0, alpha=0.3, color=colors[i])
        else: 
            ax1.plot(z,nz,color=colors[i])
            ax1.fill_between(z,nz, 0, alpha=0.3, color=colors[i])
    # Lenses
    z = sacc.tracers['lens_0'].z
    delta_z = z[1] - z[0]
    for i in np.arange(nbins_lens):
        nz = sacc.tracers[f'lens_{i}'].nz
        area = sum(delta_z * nz)
        # area normalization
        nz /= area
        if i == 0:
            ax2.plot(z,nz,color=colors[i],label='Lenses')
            ax2.fill_between(z,nz, 0, alpha=0.3, color=colors[i])
        else: 
            ax2.plot(z,nz,color=colors[i])
            ax2.fill_between(z,nz, 0, alpha=0.3, color=colors[i])
    for ax in (ax1, ax2):
        ax.set_xlim([0.0,2.5])
        ax.set_ylim([0.0,6.7])
        ax.set_ylabel('n(z)')
    ax2.set_xticks([0.5,1.0,1.5,2.0])
    ax1.text(0.7, 0.7, 'Sources',transform=ax1.transAxes,fontsize=12)
    ax2.text(0.7, 0.7, 'Lenses',transform=ax2.transAxes,fontsize=12)
    ax2.set_xlabel('redshift')
    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'dndz.png'),
                     dpi=300,
                     bbox_inches='tight')
        plt.savefig(os.path.join(savepath, 'dndz.pdf'),
                     dpi=300,
                     bbox_inches='tight')
        
    plt.show()
    plt.close()
    return()

###############################
###  Literature comparison  ###
###############################
def HikageShear_Dells():
    path = "/pscratch/sd/d/davidsan/3x2pt-HSC/pipeline_test"
    # Hikage et al. 
    # HSC data is presented as D_ell = ell * (ell + 1) /(2 * np.pi) * C_ell
    Dell = np.loadtxt(os.path.join(path, 'hsc_datavectors', 'band_powers.dat'))
    err = np.loadtxt(os.path.join(path, 'hsc_datavectors', 'band_errors.dat'))
    ell = Dell[:,0]
    return(ell, Dell, err)

def NicolaShear_Dells(i,j):
    # A. Nicola et al. Cosmic shear with HSC (Gaussian cov.)
    s_and = sacc.Sacc.load_fits('/pscratch/sd/d/davidsan/3x2pt-HSC/pipeline_test/andrina_datavectors/cls_signal_covG_HSC.fits')
    s_and_ng = sacc.Sacc.load_fits('/pscratch/sd/d/davidsan/3x2pt-HSC/pipeline_test/andrina_datavectors/cls_noise_covNG_HSC.fits')
    # Signal
    ell, C_ell_and, cov_and = s_and.get_ell_cl('cl_ee', f'wl_{i}', f'wl_{j}', return_cov=True)
    # Noise
    _, n_ell_and, cov_ng_and = s_and_ng.get_ell_cl('cl_ee', f'wl_{i}', f'wl_{j}', return_cov=True)
    # Removing noise
    # C_ell = C_ell - n_ell_and
    # Adding non-gaussian covariance
    cov_and = cov_and + cov_ng_and
    err_and = np.sqrt(np.diag(cov_and))
    pref = ell * (ell + 1) / (2 * np.pi)
    Dell = C_ell_and * pref * 10**4
    err = err_and * pref * 10**4
    return(ell, Dell, err)

def NicolaClustering_Cells(i):
    print('>>  Galaxy clustering - Nicola et al.')
   # Andrina's measurement
    path_andrina = "/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/pipeline_test/andrina_clustering_measurements/"

    cls_meas = np.load(os.path.join(path_andrina,"cls_wdpj.npy"),
                       allow_pickle=True,
                       encoding='latin1')

    cls_err = np.load(os.path.join(path_andrina,"cls_err.npy"),
                       allow_pickle=True,
                       encoding='latin1')

    ell_and = np.array([100,200,300,400,600,800,1000,1400,1800,2200,3000,3800,4600,6200,7800,9400,12600,15800])
    # Andrinas file index coversion
    if i == 0:
        j = 6
    if i == 1:
        j = 7
    if i == 2:
        j = 8
    if i == 3:
        j = -1
    # A. Nicola clustering signal    
    ell = ell_and[:len(cls_meas[j])]
    Cell = cls_meas[j]
    err = cls_err[j]
    return(ell, Cell, err)

###############################
###  Redshift distribution  ###
###############################
def HikageSource_dndz(saccfile):
    print('>>  Source n(z) - Hamana et al.')
    # Hamana/Hikage dndz
    path_dndz = '/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/redshift_distr'
    fname = os.path.join(path_dndz, 'hsc16a_cstpcf_h2020.pz.dat.txt')
    dndz = np.loadtxt(fname)
    z = dndz[:,0]
    nbins_src = 4
    for i in np.arange(nbins_src):
        nz = dndz[:,i+1]
        # Introduce it into our IVW data vector
        saccfile.add_tracer('NZ', f'source_{i}', z, nz)
        # plt.plot(z,nz)
    # plt.title('source sample')
    # plt.show()
    return(saccfile)

def TXPipeLens_dndz(saccfile):
    print('>> Lens n(z) - This work')
    # TXPipe lens sample dndz
    path = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/outputs_gama09h/'
    fname = os.path.join(path,'summary_statistics_fourier.sacc')
    # Read nz from any of our measurements
    s_lens = sacc.Sacc.load_fits(fname)
    nbins_lens = 4
    for i in np.arange(nbins_lens):
        # Extract from our original measurement
        z = s_lens.tracers[f'lens_{i}'].z
        nz = s_lens.tracers[f'lens_{i}'].nz
        # plt.plot(z, nz, label=f'z-bin {i+1}')
        # Introduce it into our IVW data vector
        saccfile.add_tracer('NZ', f'lens_{i}', z, nz)
    # plt.title('lens sample')
    # plt.show()
    return(saccfile)

###############################
###  Combined measurements  ###
###############################
def ClusteringIVW(sacc_ivw, sacc_list):
    print('>> Adding Galaxy Clustering C_ells')
    # 20 is the number of ell bins
    num_mat = [None] * 20
    den_mat = [None] * 20
    nbins_lens = 4
    # Dummy index
    m = 0
    for i in np.arange(nbins_lens):
        for j in np.arange(nbins_lens):
            if i >= j:
                k = 0
                for fname in sacc_list:
                    # print(list_fields[k])
                    s_aux = sacc.Sacc.load_fits(fname)
                    ell, Cl, cov = s_aux.get_ell_cl('galaxy_density_cl', f'lens_{i}', f'lens_{j}', return_cov=True)
                    # Remove the noise
                    noise = s_aux.get_tag("n_ell", data_type="galaxy_density_cl", tracers=(f"lens_{i}",f"lens_{i}"))
                    Cl = Cl - noise
                    var = np.diag(cov)
                    num = Cl / var
                    den = 1 / var
                    if k == 0:
                        num_mat = num
                        den_mat = den
                    else:
                        num_mat = np.vstack([num_mat, num])
                        den_mat = np.vstack([den_mat, den])
                    k = k + 1
                Cl_ivw = np.sum(num_mat, axis=0) / np.sum(den_mat, axis=0)
                print(f"bin {i+1, j+1} - negative points {sum(Cl_ivw <= 0)}")
                var_ivw = 1 / np.sum(den_mat, axis=0)
                err_ivw = np.sqrt(var_ivw)
                # Add to the IVW data vector
                for  ind in np.arange(len(ell)):
                    sacc_ivw.add_ell_cl('galaxy_density_cl', f'lens_{i}', f'lens_{j}', ell=ell[ind], x=Cl_ivw[ind])
    return(sacc_ivw)

def GammatIVW(sacc_ivw, sacc_list):
    print('>> Adding Gamma_t C_ells')
    # 20 is the number of ell bins
    num_mat = [None] * 20
    den_mat = [None] * 20
    nbins_lens = 4
    nbins_src = 4
    # Dummy index
    m = 0
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_lens):
            k = 0
            for fname in sacc_list:
                # print(list_fields[k])
                s_aux = sacc.Sacc.load_fits(fname)
                ell, Cl, cov = s_aux.get_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', return_cov=True)
                var = np.diag(cov)
                err = np.sqrt(var)
                num = Cl / var
                den = 1 / var
                if k == 0:
                    num_mat = num
                    den_mat = den
                else:
                    num_mat = np.vstack([num_mat, num])
                    den_mat = np.vstack([den_mat, den])
                k = k + 1
            Cl_ivw = np.sum(num_mat, axis=0) / np.sum(den_mat, axis=0)
            print(f"bin {i+1, j+1} - negative points {sum(Cl_ivw <= 0)}")
            for  ind in np.arange(len(ell)):
                    sacc_ivw.add_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', ell=ell[ind], x=Cl_ivw[ind])
    return(sacc_ivw)

def ShearIVW(sacc_ivw, sacc_list):
    print('>> Adding Cosmic Shear C_ells')
    # 20 is the number of ell bins
    num_mat = [None] * 20
    den_mat = [None] * 20
    nbins_src = 4
    # Dummy index
    m = 0
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i >= j:
                k = 0
                for fname in sacc_list:
                    # print(list_fields[k])
                    s_aux = sacc.Sacc.load_fits(fname)
                    ell, Cl, cov = s_aux.get_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', return_cov=True)
                    var = np.diag(cov)
                    num = Cl / var
                    den = 1 / var
                    if k == 0:
                        num_mat = num
                        den_mat = den
                    else:
                        num_mat = np.vstack([num_mat, num])
                        den_mat = np.vstack([den_mat, den])
                    k = k + 1
                Cl_ivw = np.sum(num_mat, axis=0) / np.sum(den_mat, axis=0)
                print(f"bin {i+1, j+1} - negative points {sum(Cl_ivw <= 0)}")
                for  ind in np.arange(len(ell)):
                    sacc_ivw.add_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', ell=ell[ind], x=Cl_ivw[ind])
    return(sacc_ivw)

def CovarianceIVW(sacc_ivw, sacc_list):
    print('>> Adding covariance matrix')
    # Dummy index
    k = 0
    for fname in sacc_list:
        # print(fname)
        # Data vector -- Gaussian + SSC
        s_aux = sacc.Sacc.load_fits(fname)
        # print(s.covariance.covmat.shape)
        # Remove some tracers
        s_aux.remove_selection(data_type='galaxy_shearDensity_cl_b')
        s_aux.remove_selection(data_type='galaxy_shear_cl_bb')
        s_aux.remove_selection(data_type='galaxy_shear_cl_be')
        s_aux.remove_selection(data_type='galaxy_shear_cl_eb')
        # print(s.covariance.covmat.shape)
        #sns.heatmap(s.covariance.covmat)
        det = np.linalg.det(s_aux.covariance.covmat)
        # print(f'Total covmat -- determinant = {det}')
        if det == 0:
            print('Singular total covmat')
            covmat_inv_aux = np.linalg.pinv(s_aux.covariance.covmat)
        elif det != 0:
            covmat_inv_aux = np.linalg.inv(s_aux.covariance.covmat)
        if k == 0:
            # WARNING! Using pseudo-inverse
            covmat_inv = covmat_inv_aux
        else:
            # WARNING! Using pseudo-inverse
            covmat_inv = covmat_inv + covmat_inv_aux
        # print('det =', np.linalg.det(covmat_inv))
        # sns.heatmap(np.linalg.pinv(covmat_inv))
        # plt.show()

        k = k + 1
    # Final inversion
    covmat = np.linalg.pinv(covmat_inv)
    # print('Det(covmat) = ', np.round(np.linalg.det(covmat),2))
    # sns.heatmap(covmat)
    # plt.show()
    # Adding covariance matrix to the data vector
    sacc_ivw.add_covariance(covmat)
    return(sacc_ivw)

def Areas(file):
    # Read tracer medata info
    table = h5py.File(file)
    # Extract lens counts
    counts = np.array(table['tracers']['lens_counts'])[0]
    # Extract lens density ( in gal arcmin ^ -2)
    density = np.array(table['tracers']['lens_density'])[0]
    # Compute the area 
    area = (1 / 60) ** 2 * counts / density
    return(area)

def NormalizedAreaWeights(fname):
    areas = np.array([])
    for file in fname:
        # print(file)
        a = Areas(file = file)
        # print(a, 'sq. deg.')
        # Save the area
        areas = np.append(areas, a)
        # print(areas)
        # Compute the weights 
        weights = np.copy(areas)
        # Normalize
        norm = np.sum(weights)
        weights = weights / norm
        # Check normalization
        # print(weights)
        # print(np.sum(weights))
    return(weights)

def ShearAW(sacc_aw, sacc_list, meta_list):
    # Area weighting
    print('>> Adding Area Weight. Cosmic Shear C_ells')
    nbins_src = 4
    # Compute area weights
    weights = NormalizedAreaWeights(fname = meta_list)
    # Dummy index
    m = 0
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i >= j:
                k = 0
                for fname, w in zip(sacc_list, weights):
                    # print(list_fields[k])
                    s_aux = sacc.Sacc.load_fits(fname)
                    ell, Cl, cov = s_aux.get_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', return_cov=True)
                    # Weighting process
                    Cl_aux = w * Cl
                    # Summatory
                    if k == 0:
                        Cl_aw = Cl_aux
                    else:
                        Cl_aw += Cl_aux
                    k = k + 1
                print(f"bin {i+1, j+1} - negative points {sum(Cl_aw <= 0)}")
                for  ind in np.arange(len(ell)):
                    sacc_aw.add_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', ell=ell[ind], x=Cl_aw[ind])
    return(sacc_aw)

def ClusteringAW(sacc_aw, sacc_list, meta_list):
    # Area weighting
    print('>> Adding Area Weight. Galaxy Clustering C_ells')
    nbins_lens = 4
    # Compute area weights
    weights = NormalizedAreaWeights(fname = meta_list)
    # Dummy index
    m = 0
    for i in np.arange(nbins_lens):
        for j in np.arange(nbins_lens):
            if i >= j:
                k = 0
                for fname, w in zip(sacc_list, weights):
                    # print(list_fields[k])
                    s_aux = sacc.Sacc.load_fits(fname)
                    ell, Cl, cov = s_aux.get_ell_cl('galaxy_density_cl', f'lens_{i}', f'lens_{j}', return_cov=True)
                    # Remove the noise
                    noise = s_aux.get_tag("n_ell", data_type="galaxy_density_cl", tracers=(f"lens_{i}",f"lens_{i}"))
                    Cl = Cl - noise
                    # Weighting process
                    Cl_aux = w * Cl
                    # Summatory
                    if k == 0:
                        Cl_aw = Cl_aux
                    else:
                        Cl_aw += Cl_aux
                    k = k + 1
                print(f"bin {i+1, j+1} - negative points {sum(Cl_aw <= 0)}")
                # Add to the AW data vector
                for  ind in np.arange(len(ell)):
                    sacc_aw.add_ell_cl('galaxy_density_cl', f'lens_{i}', f'lens_{j}', ell=ell[ind], x=Cl_aw[ind])
    return(sacc_aw)

def GammatAW(sacc_aw, sacc_list, meta_list):
    # Area weighting    
    print('>> Adding Area Weight. Gamma_t C_ells')
    nbins_lens = 4
    nbins_src = 4
    # Compute area weights
    weights = NormalizedAreaWeights(fname = meta_list)
    # Dummy index
    m = 0
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_lens):
            k = 0
            for fname, w in zip(sacc_list, weights):
                # print(list_fields[k])
                s_aux = sacc.Sacc.load_fits(fname)
                ell, Cl, cov = s_aux.get_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', return_cov=True)
                # Weighting process
                Cl_aux = w * Cl
                # Summatory
                if k == 0:
                    Cl_aw = Cl_aux
                else:
                    Cl_aw += Cl_aux
                k = k + 1
            print(f"bin {i+1, j+1} - negative points {sum(Cl_aw <= 0)}")
            for  ind in np.arange(len(ell)):
                    sacc_aw.add_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', ell=ell[ind], x=Cl_aw[ind])
    return(sacc_aw)

def CovarianceAW(sacc_aw, sacc_list, meta_list):
    # Area weighting    
    print('>> Adding Area Weight. covariance matrix')
    # Compute area weights
    weights = NormalizedAreaWeights(fname = meta_list)
    # Dummy index
    k = 0
    for fname, w in zip(sacc_list, weights):
        # print(fname)
        # Data vector -- Gaussian + SSC
        s_aux = sacc.Sacc.load_fits(fname)
        # print(s.covariance.covmat.shape)
        # Remove some tracers
        s_aux.remove_selection(data_type='galaxy_shearDensity_cl_b')
        s_aux.remove_selection(data_type='galaxy_shear_cl_bb')
        s_aux.remove_selection(data_type='galaxy_shear_cl_be')
        s_aux.remove_selection(data_type='galaxy_shear_cl_eb')
        # Weighted covariance matrix
        covmat_aux = w * s_aux.covariance.covmat
        # Summatory 
        if k == 0:
            covmat_weight = covmat_aux
        else:
            covmat_weight += covmat_aux
        k = k + 1
    # print('Det(covmat) = ', np.round(np.linalg.det(covmat_weight),2))
    # sns.heatmap(covmat_weight)
    # plt.show()
    # Adding covariance matrix to the data vector
    sacc_aw.add_covariance(covmat_weight)
    return(sacc_aw)

def Generate_Hikage_Shear_Cells():
    # Inputs from Hikage et al.
    # - dndz
    # - Dells
    # - Dells covariance
    # Outputs:
    # - Cells data vector
    print('<< Generating Hikage et al. cosmic shear data vector in Sacc format>>')
    # path_to_save = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw'
    path_to_save = '/pscratch/sd/d/davidsan/HSC-PDR1-3x2pt-harmonic-methods/data/harmonic/hikage/sacc/'
    # Initialize empty sacc file
    s = sacc.Sacc()
    nbins_src = 4
    #################################
    ###   Redshift distribution   ###
    #################################
    # Source sample
    # Hamana et al. / Hikage et al.
    s = HikageSource_dndz(saccfile = s)
    #################################
    ###   Angular power spectra   ###
    ###       & covariance        ###
    #################################
    path_hikage = "/pscratch/sd/d/davidsan/3x2pt-HSC/pipeline_test"
    # Hikage et al. 
    # HSC data is presented as D_ell = ell * (ell + 1) /(2 * np.pi) * C_ell
    hsc_signal = np.loadtxt(os.path.join(path_hikage, 'hsc_datavectors', 'band_powers.dat'))
    hsc_error = np.loadtxt(os.path.join(path_hikage, 'hsc_datavectors', 'band_errors.dat'))
    hsc_covariance = np.loadtxt(os.path.join(path_hikage, 'hsc_datavectors', 'cov_powers.dat'))
    ell = hsc_signal[:,0]
    pref = ell * (ell + 1) / (2 * np.pi)
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i >= j:
                # print(i,j)
                if (i == 0) and (j == 0):
                    hik_ind = 1
                elif (i == 1) and (j == 0):
                    hik_ind = 2
                elif (i == 2) and (j == 0):
                    hik_ind = 3
                elif (i == 3) and (j == 0):
                    hik_ind = 4
                elif (i == 1) and (j == 1):
                    hik_ind = 5
                elif (i == 2) and (j == 1):
                    hik_ind = 6
                elif (i == 3) and (j == 1):
                    hik_ind = 7
                elif (i == 2) and (j == 2):
                    hik_ind = 8
                elif (i == 3) and (j == 2):
                    hik_ind = 9
                elif (i == 3) and (j == 3):
                    hik_ind = 10
                # Conversion to Cell
                Cell = hsc_signal[:,hik_ind] / pref
                for  ind in np.arange(len(ell)):
                    s.add_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', ell=ell[ind], x=Cell[ind])
    # Transforming covariance from Dell to Cell
    ell_row = np.copy(ell)
    ell_column = np.copy(ell)
    # Prefactor matrix
    pref_mat = np.zeros([6,6])

    for i in np.arange(len(ell_row)):
        pref_row = ell_row[i] * (ell_row[i] + 1) / (2 * np.pi)
        for j in np.arange(len(ell_column)):
            pref_column = ell_column[j] * (ell_column[j] + 1) / (2 * np.pi)
            prefactor_aux = pref_row * pref_column
            pref_mat[i,j] = prefactor_aux
    mat = np.hstack((pref_mat,pref_mat,pref_mat,pref_mat,pref_mat,pref_mat,pref_mat,pref_mat,pref_mat,pref_mat))
    mat = np.vstack((mat,mat,mat,mat,mat,mat,mat,mat,mat,mat))
    # Removing prefactor
    hsc_covariance_Cell = hsc_covariance / mat
    # Introducing covariance
    # s.add_covariance(hsc_covariance_Cell)
    s.add_covariance(hsc_covariance_Cell)
    # Save the data vector
    s.save_fits(os.path.join(path_to_save, 'summary_statistics_fourier_Hik_shear_Cells_Hik_covmat_Hik_dndz.sacc'), overwrite=True)
    return(s)

def Generate_Hamana_Shear_CorrFunc():
    print('>> Generating Hamana et al. Sacc shear in REAL space data vector')
    nbins_src = 4
    path = '/pscratch/sd/d/davidsan/3x2pt-HSC/HSC-3x2pt-methods/real_space_data_Hamana/'

    s = sacc.Sacc()
    # Xip and xim is already in the data vector
    s = HikageSource_dndz(saccfile = s)

    ###############
    ###   Xim   ###
    ###############
    xim = np.loadtxt(os.path.join(path,'hsc16a_cstpcf_h2020.xim.txt'))
    print('>> Introducing Xi_minus')
    theta = xim[:,0]
    print('Theta = ',theta,' arcmin')
    k = 1
    # data_type='galaxy_shear_xi_minus'
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i <= j:
                print('Corr = ', i,j)
                # Read value
                val = xim[:,k]
                # print(val)
                k += 1
                # Intruce corr func value in sacc
                s.add_theta_xi(data_type='galaxy_shear_xi_minus',
                               tracer1=f'source_{i}',tracer2=f'source_{j}',
                               theta=theta,
                               x = val)
    ###############
    ###   Xip   ###
    ###############
    xip = np.loadtxt(os.path.join(path,'hsc16a_cstpcf_h2020.xip.txt'))
    print('>> Introducing Xi_plus')
    theta = xip[:,0]
    print('Theta = ',theta,' arcmin')
    k = 1
    # data_type='galaxy_shear_xi_plus'
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i <= j:
                print('Corr = ', i,j)
                # Read value
                val = xip[:,k]
                # print(val)
                k += 1
                # Intruce corr func value in sacc
                s.add_theta_xi(data_type='galaxy_shear_xi_plus',
                               tracer1=f'source_{i}',tracer2=f'source_{j}',
                               theta=theta,
                               x = val)
    print(len(s.mean))
    print('>> Building and introducing covariance')
    # Covariance matrix
    cov_array = np.loadtxt(os.path.join(path,'hsc16a_cstpcf_h2020.cov.txt'))

    n = len(s.mean)
    cov = np.zeros((n, n), dtype=float)

    k = 0
    # Read values into cov using nested loops
    for j in range(n):
        for i in range(n):
            cov[i, j] = float(cov_array[k])
            k += 1

    cov_flipped = cov[::-1,::-1]
    sns.heatmap(cov_flipped)
    s.add_covariance(cov_flipped)
    print('>> Saving Sacc data vector')
    path_to_save = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/'
    s.save_fits(os.path.join(path_to_save,'summary_statistics_real_Hamana_shear.sacc'),overwrite=True)
    return(s)

def ApplyHikageShearCuts(sacc_list):
    # Scale cuts from HSC PDR1 shear analysis
    # Hikage et al. with 300 < ell < 1900
    text_to_add = 'HikageShearSC'
    for sacc_fname in sacc_list:
        if os.path.exists(sacc_fname):
            # New filename
            root, extension = os.path.splitext(sacc_fname)
            sacc_fname_save = f"{root}_{text_to_add}{extension}"
            print(f'-{sacc_fname_save}')
            # Read sacc
            s = sacc.Sacc.load_fits(sacc_fname)
            # Apply scale cuts  # COSMIC SHEAR
            print('Number of data points before cuts: ', len(s.mean))
            print('Shape of the covariance before cuts: ', s.covariance.covmat.shape)
            s.remove_selection(data_type='galaxy_shear_cl_ee', ell__gt=1900)
            s.remove_selection(data_type='galaxy_shear_cl_ee', ell__lt=300)
            print('Number of data points after cuts: ', len(s.mean))
            print('Shape of the covariance after cuts: ', s.covariance.covmat.shape)
            # Save sacc
            print('>> >> Saving ...')
            s.save_fits(sacc_fname_save, overwrite=True)
        else:
            print('>> File does not exist')
            continue
    return()

def ApplyHamanaShearCuts(sacc_list):
    # Scale cuts from HSC PDR1 shear in real space analysis
    # Hamana et al. xip \in 7.08 < theta < 56.2 arcmin
    #               xim \in 28.2 < theta < 178.8 arcmin
    text_to_add = 'HamanaShearSC'
    for sacc_fname in sacc_list:
        if os.path.exists(sacc_fname):
            # New filename
            root, extension = os.path.splitext(sacc_fname)
            sacc_fname_save = f'{root}_{text_to_add}{extension}'
            print(f'-{sacc_fname_save}')
            # Read sacc
            s = sacc.Sacc.load_fits(sacc_fname)
            # Apply scale cuts # COSMIC SHEAR
            print('Number of data points before cuts: ', len(s.mean))
            print('Shape of the covariance before cuts: ', s.covariance.covmat.shape)
            # Xip
            s.remove_selection(data_type='galaxy_shear_xi_plus', theta__lt=7.08)
            s.remove_selection(data_type='galaxy_shear_xi_plus', theta__gt=56.2)
            # xim
            s.remove_selection(data_type='galaxy_shear_xi_minus', theta__lt=28.2)
            s.remove_selection(data_type='galaxy_shear_xi_minus', theta__gt=178.8)
            print('Number of data points after cuts: ', len(s.mean))
            print('Shape of the covariance after cuts: ', s.covariance.covmat.shape)
            # Save sacc
            print('>> >> Saving ...')
            s.save_fits(sacc_fname_save, overwrite=True)
        else:
            print('>> File does not exist')
            continue
    return(s)
    
def Generate_TXPipe_CombMeas_Cells(sacc_list, meta_list, combmethod, path_to_save, label, shear_cuts=True):
    # combmethod = 'ivw' or 'aw'
    print('<< Combined TXPipe data vector generation >>')
    # path_to_save = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw'
    # Initialize empty sacc file
    s = sacc.Sacc()
    # Hard-wired numbers, HSC specific
    nbins_lens = 4
    nbins_src = 4
    #################################
    ###   Redshift distribution   ###
    #################################
    # Source sample
    # Hamana et al. / Hikage et al.
    s = HikageSource_dndz(saccfile = s)
    # Lens sample
    s = TXPipeLens_dndz(saccfile = s)
    #################################
    ###   Angular power spectra   ###
    ###       & covariance        ###
    #################################
    # Angular power spectra 3x2pt
    if combmethod == 'ivw':
        print('>> Combining following Inverse Variance Method')
        # Cells - IVW combination of the individual fields
        # Clustering
        s = ClusteringIVW(sacc_ivw = s, sacc_list = sacc_list) 
        # Gglensing
        s = GammatIVW(sacc_ivw = s, sacc_list = sacc_list)
        # Shear
        s = ShearIVW(sacc_ivw = s, sacc_list = sacc_list)
        # Covariance matrix - IVW
        # Cov^{-1}_{total} = sum_{i = fields} Cov^{-1}_{i}
        s = CovarianceIVW(sacc_ivw = s, sacc_list = sacc_list)
        # Save the data vector
        s.save_fits(os.path.join(path_to_save, 'summary_statistics_fourier_ivw.sacc'), overwrite=True)
    elif combmethod == 'aw':
        print('>> Combining following Area Weighting Method')
        # Cells - Area weighting of the individual fields
        # Clustering
        s = ClusteringAW(sacc_aw = s, sacc_list = sacc_list, meta_list = meta_list)
        # Gglensing
        s = GammatAW(sacc_aw = s, sacc_list = sacc_list, meta_list = meta_list)
        # Shear
        s = ShearAW(sacc_aw = s, sacc_list = sacc_list, meta_list = meta_list)
        # Covariance matrix - AW
        # Cov_{total} = sum(i = fields) w_i * Cov_i
        s = CovarianceAW(sacc_aw = s, sacc_list = sacc_list, meta_list = meta_list)
        s.save_fits(os.path.join(path_to_save, 'summary_statistics_fourier_aw.sacc'), overwrite=True)
        
    if shear_cuts:
        print('>> Applying Hikage et al. shear cuts (300 < ell < 1900)')
        # Add to individual field sacc, IVW and AW measurements
        sacc_list = np.append(sacc_list, os.path.join(path_to_save, f'summary_statistics_fourier_ivw_{label}.sacc'))
        sacc_list = np.append(sacc_list, os.path.join(path_to_save, f'summary_statistics_fourier_aw_{label}.sacc'))
        # Application
        ApplyHikageShearCuts(sacc_list = sacc_list)
    return(s)

def Read_TXPipe_CombMeas_Cells(probe,i,j,combmethod):
    # shear 'galaxy_shear_cl_ee'
    # Read our IVW combined measuremtent
    if combmethod == 'ivw':
        fname = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_fourier_ivw.sacc'
    elif combmethod == 'aw':
        fname = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_fourier_aw.sacc'
    else:
        print('Combination method does not exist!')
    # Read sacc
    s = sacc.Sacc.load_fits(fname)
    # Extract Cls
    if probe == 'galaxy_shear_cl_ee':
        ell, Cell, cov = s.get_ell_cl(probe, f'source_{i}', f'source_{j}', return_cov=True)
    elif probe == 'galaxy_density_cl':
        ell, Cell, cov = s.get_ell_cl(probe, f'lens_{i}', f'lens_{j}', return_cov=True)
    elif probe == 'galaxy_shearDensity_cl_e':
        ell, Cell, cov = s.get_ell_cl(probe, f'source_{i}', f'lens_{j}', return_cov=True)
    # extracting error from covariance
    err = np.sqrt(np.diag(cov))
    return(ell, Cell, err, cov)

################################
###  Plotting 2pt functions  ###
################################

def Shear2pt_plot(fname,labels,add_individual=False, add_combined=False, add_literature=False, add_Hikage_sacc=False, theory_fname=None, just_auto = False, save_fig=False):
    # fname list of sacc data vectors
    # labels list of labels for the dvs
    # add_literature Add Hikage et al. and Nicola et al. measurements
    # save_fig
        
    nbins_src = 4
    # generate the subplot structure
    if just_auto:
        fig, axs = plt.subplots(1, nbins_src, sharex=True, sharey='row', figsize=(10,3))
    else:
        fig, axs = plt.subplots(nbins_src, nbins_src, sharex=True, sharey='row', figsize=(10,10))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Initialize text label for savefig
    textfig = 'Shear2pt'
    # Initialize figure and format
    #loop over redshift bins
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_src):
            if i >= j:
                axind = (i,j)
                if just_auto:
                    if i == j:
                        axind = i
                        pass
                    else:
                        continue
                # log-log scale
                axs[axind].set_xscale('log')
                # scale cuts
                axs[axind].axvline(300, ls='--', c='k', linewidth=0.5)
                axs[axind].axvline(1900, ls='--', c='k', linewidth=0.5)
                axs[axind].axhline(0, ls='--', c='k', linewidth=0.5)
                axs[axind].axvspan(xmin=50, xmax=300, color='grey', alpha=0.01)
                axs[axind].axvspan(xmin=1900, xmax=6800, color='grey', alpha=0.01)
                # x-lim range
                axs[axind].set_xlim([90, 6144])
                # z-bin pair
                axs[axind].text(0.15, 0.85,f'({i + 1},{j + 1})', ha='center', va='center', transform=axs[axind].transAxes, fontsize=12)
                # y-lims
                if i == 0:
                    axs[axind].set_ylim([-0.2, 0.6])
                if i == 1:
                    axs[axind].set_ylim([-0.2, 1.2])
                if i == 2:
                    axs[axind].set_ylim([-0.5, 2.2])
                if i == 3:
                    axs[axind].set_ylim([-0.5, 2.7])
                if j == 0:
                    axs[axind].set_ylabel('$\mathcal{D}^{\kappa \kappa}_\ell [\\times 10^4]$')
                if just_auto:
                    axs[axind].set_xlabel('multipole, $\ell$')
                    axs[axind].set_xticks((100,1000),labels=('100','1000'))
                else:
                    if (i == nbins_src - 1):
                        axs[axind].set_xlabel('multipole, $\ell$')
                        axs[axind].set_xticks((100,1000),labels=('100','1000'))
            else:
                if just_auto:
                    continue
                else:
                    axs[i,j].axis('off')
    
    if add_individual == True:
        textfig += '_Fields' 
        ##############################
        ###   Fields measurements  ###
        ##############################
        k = 1
        # loop over the 6 different fields
        for fn, lab in zip(fname,labels):
            # read Sacc file
            s = sacc.Sacc.load_fits(fn)
            #loop over redshift bins
            for i in np.arange(nbins_src):
                for j in np.arange(nbins_src):
                    if i >= j:
                        axind = (i,j)
                        if just_auto:
                            if i == j:
                                axind = i
                                pass
                            else:
                                continue
                        # read cosmic shear data points
                        ell, Cell, cov = s.get_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', return_cov=True)
                        # extracting error from covariance
                        err = np.sqrt(np.diag(cov))
                        # computing prefactor 
                        pref = ell * (ell + 1) / (2 * np.pi)
                        # compute Dell = l * (l + 1) * Cell / (2 * pi)
                        Dell = pref * Cell * 10 ** 4
                        err = pref * err * 10 ** 4
                        # plot
                        axs[axind].errorbar(ell, Dell, err, 
                                          # color=colors[k], 
                                          fmt='o', 
                                          markersize=3.0, 
                                          capsize=2,
                                          alpha=1.0,
                                          label=f'{lab}')
                                          # label=f'{lab.upper()}')
            # new color for the next field
            k += 1

    if add_literature == True:
        textfig += '_Literature'
        # Read Hikage et al.
        print('>>  Cosmic shear - Hikage et al.')
        ell_hik, Dell_hik, err_hik = HikageShear_Dells()
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_src):
                if i >= j:
                    axind = (i,j)
                    if just_auto:
                        if i == j:
                            axind = i
                            pass
                        else:
                            continue
                    ###############################################################
                    # Hikage et al.
                    # mask = (ell > 300)*(ell < 1900)
                    if (i == 0) and (j == 0):
                        hik_ind = 1
                    elif (i == 1) and (j == 0):
                        hik_ind = 2
                    elif (i == 2) and (j == 0):
                        hik_ind = 3
                    elif (i == 3) and (j == 0):
                        hik_ind = 4
                    elif (i == 1) and (j == 1):
                        hik_ind = 5
                    elif (i == 2) and (j == 1):
                        hik_ind = 6
                    elif (i == 3) and (j == 1):
                        hik_ind = 7
                    elif (i == 2) and (j == 2):
                        hik_ind = 8
                    elif (i == 3) and (j == 2):
                        hik_ind = 9
                    elif (i == 3) and (j == 3):
                        hik_ind = 10
                    Dell = Dell_hik[:,hik_ind] * 10**4
                    err = err_hik[:,hik_ind] * 10**4
                    # print('Error Hikage')
                    axs[axind].errorbar(ell_hik, Dell, yerr=err, 
                                        c=colors[7],
                                        fmt='o', 
                                        markersize=3.0, 
                                        capsize=2,
                                        label='Hikage et al.')
                    ###############################################################
        # Read Nicola et al.
        print('>>  Cosmic shear - Nicola et al.')
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_src):
                if i <= j:
                    axind = (j,i)
                    if just_auto:
                        if i == j:
                            axind = i
                            pass
                        else:
                            continue
                    #############################################################
                    # A. Nicola et al. Cosmic shear with HSC (Gaussian cov.)
                    ell_and, Dell_and, err_and = NicolaShear_Dells(i,j)
                    axs[axind].errorbar(ell_and, Dell_and, yerr=err_and, 
                                          c=colors[8],
                                          fmt='o', 
                                          markersize=3.0, 
                                          capsize=2,
                                          label='Nicola et al.')
                    ###############################################################
    if add_Hikage_sacc == True:
        textfig += '_HikageSacc'
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_src):
                if i <= j:
                    axind = (j,i)
                    if just_auto:
                        if i == j:
                            axind = i
                            pass
                        else:
                            continue
                    # fname_hik = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_fourier_Hik_shear_Cells_Hik_covmat_Hik_dndz.sacc'
                    fname_hik = '/global/cfs/projectdirs/lsst/groups/LSS/HSC_reanalysis/data_javi/cls_hscpdr1_hikage_wcov_cholesky.sacc'
                    # Covariance * 2pi
                    # fname_hik = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_fourier_Hik_shear_Cells_Hik_2pi_times_covmat_Hik_dndz.sacc'
                    s_hik = sacc.Sacc.load_fits(fname_hik)
                    # Read IVW combined measurement (This work)
                    ell_hik_sacc, Cell_hik_sacc, cov_hik_sacc = s_hik.get_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', return_cov=True)
                    err_hik_sacc = np.sqrt(np.diag(cov_hik_sacc))
                    # computing prefactor 
                    pref = ell_hik_sacc * (ell_hik_sacc + 1) / (2 * np.pi)
                    # compute Dell = l * (l + 1) * Cell / (2 * pi)
                    Dell_hik_sacc = pref * Cell_hik_sacc * 10 ** 4
                    err_hik_sacc = pref * err_hik_sacc * 10 ** 4

                    # plot
                    axs[axind].errorbar(ell_hik_sacc, Dell_hik_sacc, err_hik_sacc, 
                                      color='green', 
                                      fmt='o', 
                                      markersize=3.0, 
                                      capsize=2,
                                      label='Hikage et al. (Sacc)')
    if add_combined is not None:
        if add_combined == 'aw':
            textfig += '_CombinedAW'
            for i in np.arange(nbins_src):
                for j in np.arange(nbins_src):
                    if i >= j:
                        axind = (i,j)
                        if just_auto:
                            if i == j:
                                axind = i
                                pass
                            else:
                                continue
                        ######################
                        ### Area Weighting ###
                        ######################
                        # Read IVW combined measurement (This work)
                        ell_txp, Cell_txp, err_txp, cov_txp = Read_TXPipe_CombMeas_Cells(probe='galaxy_shear_cl_ee',i=i,j=j,combmethod='aw')
                        # computing prefactor 
                        pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                        # compute Dell = l * (l + 1) * Cell / (2 * pi)
                        Dell_txp = pref * Cell_txp * 10 ** 4
                        err_txp = pref * err_txp * 10 ** 4

                        # plot
                        axs[axind].errorbar(ell_txp, Dell_txp, err_txp, 
                                          color='orange', 
                                          fmt='o', 
                                          markersize=3.0, 
                                          capsize=2,
                                          label='This work (AW)')
        elif add_combined == 'ivw':
            textfig += '_CombinedIVW'
            for i in np.arange(nbins_src):
                for j in np.arange(nbins_src):
                    if i >= j:
                        axind = (i,j)
                        if just_auto:
                            if i == j:
                                axind = i
                                pass
                            else:
                                continue
                        ##################################
                        ### Inverse Variance Weighting ###
                        ##################################
                        # Read IVW combined measurement (This work)
                        ell_txp, Cell_txp, err_txp, cov_txp = Read_TXPipe_CombMeas_Cells(probe='galaxy_shear_cl_ee',i=i,j=j,combmethod='ivw')
                        # computing prefactor 
                        pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                        # compute Dell = l * (l + 1) * Cell / (2 * pi)
                        Dell_txp = pref * Cell_txp * 10 ** 4
                        err_txp = pref * err_txp * 10 ** 4

                        # plot
                        axs[axind].errorbar(ell_txp, Dell_txp, err_txp, 
                                          color=colors[0], 
                                          fmt='o', 
                                          markersize=3.0, 
                                          capsize=2,
                                          label='This work')
                        
                        snr = ComputeSNR(signal = Cell_txp, cov = cov_txp)
                        # z-bin pair
                        axs[axind].text(0.15, 0.10,f'S/N = {np.round(snr,2)}', ha='center', va='center', transform=axs[axind].transAxes, fontsize=6)
    if theory_fname is not None:
        textfig += '_Theory'
        # Adding chisq info
        npar = 12
        chisq, chisq_ndof, ndof = ComputeChisq(sacc_fname = "/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_fourier_ivw_HikageShearSC.sacc",
             theory_fname = theory_fname,
             probe = 'galaxy_shear_cl_ee',
             npar = npar)
        text = f'$\chi^2 / \\nu = {np.round(chisq, 2)} / {int(ndof)}$ = {np.round(chisq_ndof, 2)}'
        # axs[0,0].text(0.85, 0.85,f'$\chi^2 / \\nu = {np.round(chisq, 2)} / {int(npar)}$ = {np.round(chisq_ndof, 2)}', ha='center', va='center', transform=axs[0,0].transAxes, fontsize=8)
        for i in range(nbins_src):
            for j in range(i, nbins_src):
                if i <= j:
                    axind = (j,i)
                    if just_auto:
                        if i == j:
                            axind = i
                            pass
                        else:
                            continue
                # read cosmic shear data points
                ell_th = np.loadtxt(os.path.join(theory_fname, f'ell_or_theta_galaxy_shear_cl_ee_source_{j}_source_{i}.txt'))
                Cell_th = np.loadtxt(os.path.join(theory_fname,f'theory_galaxy_shear_cl_ee_source_{j}_source_{i}.txt'))
                # computing prefactor 
                pref = ell_th * (ell_th + 1) / (2 * np.pi)
                # compute Dell = l * (l + 1) * Cell / (2 * pi)
                Dell_th = pref * Cell_th * 10 ** 4
                # plot
                axs[axind].plot(ell_th, Dell_th, lw=1.2, color='k', label = text)
    
    if just_auto:
        plt.legend(bbox_to_anchor=(0.0, 1.05),ncol=3,frameon=False,fontsize=12)
    else:
        plt.legend(bbox_to_anchor=(0.0, 2.45),frameon=False,fontsize=12)
    if save_fig == True:
        print('>> Saving figure ...')
        print(textfig)
        plt.savefig(f'{textfig}.png',
                    dpi=300,
                    bbox_inches='tight')
        plt.savefig(f'{textfig}.pdf',
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()
    return()
def Shear2pt_plot_Hamana_real(save_fig=False):
    nbins_src = 4
    # Xip
    fig_xip, axs_xip = plt.subplots(nbins_src,nbins_src,sharex=True,sharey='row',figsize=(8,8))
    fig_xip.tight_layout()
    # xim
    fig_xim, axs_xim = plt.subplots(nbins_src,nbins_src,sharex=True,sharey='row',figsize=(8,8))
    fig_xim.tight_layout()

    fig_xip.subplots_adjust(wspace=0, hspace=0)
    fig_xim.subplots_adjust(wspace=0, hspace=0)

    # David
    # fname = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_real_Hamana_shear.sacc'
    # Emily's and Chihway
    fname = '/pscratch/sd/d/davidsan/txpipe-reanalysis/hsc/outputs/ivw/summary_statistics_real_raw_hsc_HamanaShearSC.sacc'
    s = sacc.Sacc.load_fits(fname)
    # Extract the errors
    error = np.sqrt(np.diag(s.covariance.covmat))# [::-1]

    axs_xip[0,0].axis('off')
    axs_xip[0,1].axis('off')
    axs_xip[0,2].axis('off')
    axs_xip[1,0].axis('off')
    axs_xip[1,1].axis('off')
    axs_xip[2,0].axis('off')

    axs_xim[0,0].axis('off')
    axs_xim[0,1].axis('off')
    axs_xim[0,2].axis('off')
    axs_xim[1,0].axis('off')
    axs_xim[1,1].axis('off')
    axs_xim[2,0].axis('off')

    axs_xip[3,0].set_ylabel('$\\theta \\xi_+ \\times 10^{4}$')
    axs_xip[3,0].set_xlabel('$\\theta$ [arcmin]')

    axs_xim[3,0].set_ylabel('$\\theta \\xi_- \\times 10^{4}$')
    axs_xim[3,0].set_xlabel('$\\theta$ [arcmin]')

    # Error index
    k = 0
    for i in np.arange(1,5):
        for j in np.arange(1,5):    
            if  i <= j:
                print('Bin ', i,j)
                if j == 1 and i == 1:
                    axind = (3,0)
                    axs_xip[axind].set_ylim([-0.3,2.3])
                    axs_xim[axind].set_ylim([-0.3,2.3])
                    axs_xip[axind].set_yticks([0,0.5,1.0,1.5,2.0])
                    axs_xim[axind].set_yticks([0,0.5,1.0,1.5,2.0])
                elif j == 2 and i == 1:
                    axind = (3,1)
                elif j == 3 and i == 1:
                    axind = (3,2)
                elif j == 4 and i == 1:
                    axind = (3,3)
                elif j == 2 and i == 2:
                    axind = (2,1)
                    axs_xip[axind].set_ylim([0.0,4.4])
                    axs_xim[axind].set_ylim([0.0,4.4])
                    axs_xip[axind].set_yticks([1,2,3,4])
                    axs_xim[axind].set_yticks([1,2,3,4])
                elif j == 3 and i == 2:
                    axind = (2,2)
                elif j == 4 and i == 2:
                    axind = (2,3)
                elif j == 3 and i == 3:
                    axind = (1, 2)
                    axs_xip[axind].set_ylim([0.0, 7.0])
                    axs_xim[axind].set_ylim([0.0, 7.0])
                    axs_xip[axind].set_yticks([2,4,6])
                    axs_xim[axind].set_yticks([2,4,6])
                elif j == 4 and i == 3:
                    axind = (1,3)
                elif j == 4 and i == 4:
                    axind = (0,3)
                    axs_xip[axind].set_ylim([0.0,10.0])
                    axs_xim[axind].set_ylim([0.0,10.0])
                    axs_xip[axind].set_yticks([2,4,6,8])
                    axs_xim[axind].set_yticks([2,4,6,8])
                ###############
                ###   Xim   ###
                ###############
                th,xim=s.get_theta_xi(data_type='galaxy_shear_xi_minus', tracer1=f'source_{i}',tracer2=f'source_{j}')
                # print(xip)
                err_xim = error[k: k + len(xim)]
                k += len(xim)
                # print('len xip', len(xip))
                # print(k)
                xim *= th * 10 ** 4
                err_xim *= th * 10 ** 4
                axs_xim[axind].errorbar(th,xim,err_xim,
                                        color='red',
                                        fmt='o', 
                                        markersize=3.0, 
                                        capsize=2,
                                        label = 'Hamana et al.')
                axs_xim[axind].axhline(y=0,ls='dotted',color='k')
                axs_xim[axind].axvline(x=28.2,ls='dotted',color='k')
                axs_xim[axind].axvline(x=178,ls='dotted',color='k')
                axs_xim[axind].set_xscale('log')
                axs_xim[axind].text(0.15, 0.85,f'({i},{j})', ha='center', va='center', transform=axs_xim[axind].transAxes,fontsize=10)
                ###############
                ###   Xip   ###
                ###############
                th,xip=s.get_theta_xi(data_type='galaxy_shear_xi_plus', tracer1=f'source_{i}',tracer2=f'source_{j}')
                # print(xip)
                err_xip = error[k: k + len(xip)]
                k += len(xip)
                # print('len xip', len(xip))
                # print(k)
                xip *= th * 10 ** 4
                err_xip *= th * 10 ** 4
                axs_xip[axind].errorbar(th,xip,err_xip,
                                        color='red',
                                        fmt='o', 
                                        markersize=3.0, 
                                        capsize=2,
                                        label = 'Hamana et al.')
                axs_xip[axind].axhline(y=0,ls='dotted',color='k')
                axs_xip[axind].axvline(x=7.08,ls='dotted',color='k')
                axs_xip[axind].axvline(x=56.2,ls='dotted',color='k')
                axs_xip[axind].set_xscale('log')
                axs_xip[axind].text(0.15, 0.85,f'({i},{j})', ha='center', va='center', transform=axs_xip[axind].transAxes,fontsize=10)
                ##################
                ###   Format   ###
                ##################
                axs_xip[i-1,j-1].set_xlim([5,400])
                axs_xim[i-1,j-1].set_xlim([5,400])
                axs_xip[i-1,j-1].set_xticks(list([10, 100]),labels=list(['10','100']))
                axs_xim[i-1,j-1].set_xticks(list([10, 100]),labels=list(['10','100']))
            
    axs_xip[0,3].legend(frameon=False,fontsize=7)
    if save_fig == True:
        fig_xip.savefig(f'Shear2pt_Xip_corrfunc_Hamana.png',
                    dpi=300,
                    bbox_inches='tight')
        fig_xip.savefig(f'Shear2pt_Xip_corrfunc_Hamana.pdf',
                    dpi=300,
                    bbox_inches='tight')
        fig_xim.savefig(f'Shear2pt_Xim_corrfunc_Hamana.png',
                    dpi=300,
                    bbox_inches='tight')
        fig_xim.savefig(f'Shear2pt_Xim_corrfunc_Hamana.pdf',
                    dpi=300,
                    bbox_inches='tight')
    return()
def Clustering2pt_plot(fname,labels,add_individual=False,add_combined=False,add_literature=False,show_residual=False,save_fig=False):
    # fname list of sacc data vectors
    # labels list of labels for the dvs
    # add_literature Add Hikage et al. and Nicola et al. measurements
    nbins_lens = 4
    # generate the subplot structure
    if show_residual == True:
        # Initialize figure to make the residual plot, the second raw must be the same width as the first one
        # but one third of the height
        fig, axs = plt.subplots(2, nbins_lens, sharex=True, sharey='row', figsize=(10,4), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, axs = plt.subplots(1, nbins_lens, sharex=True, sharey='row', figsize=(10,4))
    # fig.tight_layout()

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    # Initialize figure and format
    #loop over redshift bins
    for i in np.arange(nbins_lens):
        for j in np.arange(nbins_lens):
            if i == j:
                if show_residual == True:
                    ind_plot = (0,i)
                    ind_res =  (1,i)
                else:
                    ind_plot = i
                # log-log scale
                axs[ind_plot].set_xscale('log')
                axs[ind_plot].axhline(0, ls='--', c='k', linewidth=0.5)
                # x-lim range
                axs[ind_plot].set_xlim([90, 5500])
                axs[ind_plot].set_box_aspect(1)

                # z-bin pair
                axs[ind_plot].text(0.15, 0.85,f'({i + 1},{j + 1})', ha='center', va='center', transform=axs[ind_plot].transAxes, fontsize=12)
                if i == 0:
                    axs[ind_plot].set_ylabel('$\mathcal{D}^{\delta \delta}_\ell [\\times 10]$')
                if show_residual == True:
                    axs[ind_res].set_xlabel('multipole, $\ell$')
                    if i == 0:
                        axs[ind_res].set_ylabel('$\Delta C^{\delta \delta}_\ell / \sigma$')
                    axs[ind_res].set_xticks((100,1000),labels=('100','1000'))
                    axs[ind_res].set_ylim([-10,10])
                    axs[ind_res].axhline(0, ls='--', c='k', linewidth=0.5)
                else:
                    axs[ind_plot].set_xlabel('multipole, $\ell$')
                    axs[ind_plot].set_xticks((100,1000),labels=('100','1000'))
    if add_individual == True:
        ##############################
        ###   Fields measurements  ###
        ##############################
        k = 1
        # loop over the 6 different fields
        for fn, lab in zip(fname,labels):
            print('>> Plotting ', lab)
            # read Sacc file
            s = sacc.Sacc.load_fits(fn)
            #loop over redshift bins
            for i in np.arange(nbins_lens):
                for j in np.arange(nbins_lens):
                    if i == j:
                        if show_residual == True:
                            ind_plot = (0,i)
                            ind_res =  (1,i)
                        else:
                            ind_plot = i
                        # read cosmic shear data points
                        if 'summary' in fn:
                            if i == 0:
                                print('DV with covariance')
                            ell, Cell, cov = s.get_ell_cl("galaxy_density_cl", f'lens_{i}', f'lens_{j}', return_cov=True)
                        elif 'twopoint' in fn:
                            if i == 0:
                                print('DV w/o covariance')
                            ell, Cell = s.get_ell_cl("galaxy_density_cl", f'lens_{i}', f'lens_{j}', return_cov=False)
                        noise = s.get_tag("n_ell", data_type="galaxy_density_cl", tracers=(f"lens_{i}",f"lens_{j}"))
                        if noise is not None:
                            # Substract noise
                            print('Substracting noise')
                            Cell = Cell - noise
                        else:
                            print('No noise to substract')
                        # computing prefactor 
                        pref = ell * (ell + 1) / (2 * np.pi)
                        Dell = pref * Cell * 10
                        # print(ell,Dell)
                        # extracting error from covariance
                        if 'summary' in fn:
                            err = np.sqrt(np.diag(cov))
                            err = pref * err * 10
                            # plot
                            axs[ind_plot].errorbar(ell, Dell, err, 
                                            color=colors[k], 
                                            fmt='o', 
                                            markersize=3.0, 
                                            capsize=2,
                                            alpha=0.3,
                                            label=f'{lab.upper()}')
                        elif 'twopoint' in fn:
                            axs[ind_plot].scatter(ell, Dell,
                                           color=colors[k+1],
                                           s=2.5,
                                           alpha=1.0,
                                           label=f'{lab.upper()}')
                        # Show residuals
                        if show_residual == True:
                            # Extract Nicola et al. Cell and error for this correlation
                            ell_an, Cell_an, err_an = NicolaClustering_Cells(i)
                            # Check if Cell, Cell_an and err_an are the same length if not add zeros at the end of the shorter array
                            if len(ell) != len(ell_an):
                                if len(ell) > len(ell_an):
                                    ell_an = np.append(ell_an, np.zeros(len(ell) - len(ell_an)))
                                    Cell_an = np.append(Cell_an, np.zeros(len(ell) - len(Cell_an)))
                                    err_an = np.append(err_an, np.zeros(len(ell) - len(err_an)))
                                elif len(ell) < len(ell_an):
                                    ell = np.append(ell, np.zeros(len(ell_an) - len(ell)))
                                    Cell = np.append(Cell, np.zeros(len(ell_an) - len(Cell)))
                                    err = np.append(err, np.zeros(len(ell_an) - len(err)))
                            # Compute residuals wrt to Nicola et al.
                            res = (Cell - Cell_an) / err_an
                            axs[ind_res].scatter(ell, res,
                                                color=colors[k],
                                                # Scatter are crosses
                                                marker='x',
                                                s=2.7)
            # new color for the next field
            k += 1
    if add_combined is not None:
        if add_combined == 'aw':
            #loop over redshift bins
            for i in np.arange(nbins_lens):
                for j in np.arange(nbins_lens):
                    if i == j:
                        if show_residual == True:
                            ind_plot = (0,i)
                            ind_res =  (1,i)
                        else:
                            ind_plot = i
                        # Read IVW combined measurement (This work)
                        ell_txp, Cell_txp, err_txp, cov_txo = Read_TXPipe_CombMeas_Cells(probe='galaxy_density_cl',i=i,j=j, combmethod='aw')
                        # computing prefactor 
                        pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                        # compute Dell = l * (l + 1) * Cell / (2 * pi)
                        Dell_txp = pref * Cell_txp * 10
                        err_txp = pref * err_txp * 10 

                        # plot
                        axs[ind_plot].errorbar(ell_txp, Dell_txp, err_txp, 
                                                color='orange', 
                                                fmt='o', 
                                                markersize=3.0, 
                                                capsize=2,
                                                label='This work (AW)')
        elif add_combined == 'ivw':
            #loop over redshift bins
            for i in np.arange(nbins_lens):
                for j in np.arange(nbins_lens):
                    if i == j:
                        if show_residual == True:
                            ind_plot = (0,i)
                            ind_res =  (1,i)
                        else:
                            ind_plot = i
                        # Read IVW combined measurement (This work)
                        ell_txp, Cell_txp, err_txp, cov_txp = Read_TXPipe_CombMeas_Cells(probe='galaxy_density_cl',i=i,j=j, combmethod='ivw')
                        # computing prefactor 
                        pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                        # compute Dell = l * (l + 1) * Cell / (2 * pi)
                        Dell_txp = pref * Cell_txp * 10
                        err_txp = pref * err_txp * 10 

                        # plot
                        axs[ind_plot].errorbar(ell_txp, Dell_txp, err_txp, 
                                        color=colors[0], 
                                        fmt='o', 
                                        markersize=3.0, 
                                        capsize=2,
                                        label='This work')
                        snr = ComputeSNR(signal = Cell_txp, cov = cov_txp)
                        # z-bin pair
                        axs[i].text(0.85, 0.1,f'S/N = {np.round(snr,2)}', ha='center', va='center', transform=axs[i].transAxes, fontsize=6)
    if add_literature == True:
        for i in np.arange(nbins_lens):
            if show_residual == True:
                ind_plot = (0,i)
                ind_res =  (1,i)
            else:
                ind_plot = i
            ell_an, Cell_an, err = NicolaClustering_Cells(i)
            pref = ell_an * (ell_an + 1) / (2 * np.pi)
            Dell_an = pref * Cell_an * 10
            err_an = pref * err * 10

            # plot
            axs[ind_plot].errorbar(ell_an, Dell_an, err_an, 
                            color=colors[1], 
                            fmt='o', 
                            markersize=3.0, 
                            capsize=2,
                            label='Nicola et al.')
    """ if add_individual == False:
        if show_residual == True:
            axs[(0,3)].legend(frameon=False,fontsize=8)
        else:
            axs[3].legend(frameon=False,fontsize=8)
    else: """
    axs[ind_plot].legend(frameon=False,fontsize=6)
    if save_fig == True:
        plt.savefig('Clustering2pt.png',
            dpi=300,
            bbox_inches='tight')
        plt.savefig('Clustering2pt.pdf',
            dpi=300,
            bbox_inches='tight')
    plt.show()
    plt.close()
    return()

def Gammat2pt_plot(fname,labels,add_individual=False, add_combined=False,theory_fname=None,save_fig=False):
    # fname list of sacc data vectors
    # labels list of labels for the dvs
    nbins_lens = 4
    nbins_src = 4
    # generate the subplot structure
    fig, axs = plt.subplots(nbins_src, nbins_lens, sharex=True, sharey='row', figsize=(10,10))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # Initialize figure and format
    #loop over redshift bins
    #loop over redshift bins
    for i in np.arange(nbins_src):
        for j in np.arange(nbins_lens):
            # log-log scale
            axs[i, j].set_xscale('log')
            # # scale cuts
            axs[i,j].axhline(0, ls='--', c='k', linewidth=0.5)
            # x-lim range
            axs[i,j].set_xlim([90, 2500])
            # z-bin pair
            axs[i,j].text(0.15, 0.85,f'({i + 1},{j + 1})', ha='center', va='center', transform=axs[i,j].transAxes, fontsize=12)
            if j == 0:
                axs[i,j].set_ylabel('$\mathcal{D}^{\kappa \delta}_\ell \, [\\times 10^3]$')
            if (i == nbins_src - 1):
                axs[i,j].set_xlabel('multipole, $\ell$')
                axs[i,j].set_xticks((100,1000),labels=('100','1000'))
    if add_individual == True:
        ##############################
        ###   Fields measurements  ###
        ##############################
        k = 1
        # loop over the 6 different fields
        for fn, lab in zip(fname,labels):
            # read Sacc file
            s = sacc.Sacc.load_fits(fn)
            #loop over redshift bins
            for i in np.arange(nbins_src):
                for j in np.arange(nbins_lens):
                    ell, Cell, cov = s.get_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', return_cov=True)
                    # extracting error from covariance
                    err = np.sqrt(np.diag(cov))
                    # computing prefactor 
                    pref = ell * (ell + 1) / (2 * np.pi)
                    # compute Dell = l * (l + 1) * Cell / (2 * pi)
                    Dell = pref * Cell * 1000
                    err = pref * err * 1000

                    # plot
                    axs[i,j].errorbar(ell, Dell, err, 
                                      color=colors[k], 
                                      fmt='o', 
                                      markersize=3.0, 
                                      capsize=2,
                                      alpha=0.3,
                                      label=f'{lab.upper()}')
            # new color for the next field
            k += 1
    if add_combined is not None:
        if add_combined == 'ivw':
            #loop over redshift bins
             for i in np.arange(nbins_src):
                for j in np.arange(nbins_lens):
                    # Inverse variance weighting measurement
                    ell_txp, Cell_txp, err_txp, cov_txp = Read_TXPipe_CombMeas_Cells(probe='galaxy_shearDensity_cl_e',i=i,j=j,combmethod='ivw')
                    # computing prefactor 
                    pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                    # compute Dell = l * (l + 1) * Cell / (2 * pi)
                    Dell_txp = pref * Cell_txp * 10 ** 3
                    err_txp = pref * err_txp * 10 ** 3
                    # plot
                    axs[i,j].errorbar(ell_txp, Dell_txp, err_txp, 
                                      color=colors[0], 
                                      fmt='o', 
                                      markersize=3.0, 
                                      capsize=2,
                                      label='This work')
                    snr = ComputeSNR(signal = Cell_txp, cov = cov_txp)
                    # z-bin pair
                    axs[i,j].text(0.15, 0.70,f'S/N = {np.round(snr,2)}', ha='center', va='center', transform=axs[i,j].transAxes, fontsize=6)
        elif add_combined == 'aw':
            #loop over redshift bins
             for i in np.arange(nbins_src):
                for j in np.arange(nbins_lens):
                    # Inverse variance weighting measurement
                    ell_txp, Cell_txp, err_txp, cov_txp = Read_TXPipe_CombMeas_Cells(probe='galaxy_shearDensity_cl_e',i=i,j=j,combmethod='aw')
                    # computing prefactor 
                    pref = ell_txp * (ell_txp + 1) / (2 * np.pi)
                    # compute Dell = l * (l + 1) * Cell / (2 * pi)
                    Dell_txp = pref * Cell_txp * 10 ** 3
                    err_txp = pref * err_txp * 10 ** 3
                    # plot
                    axs[i,j].errorbar(ell_txp, Dell_txp, err_txp, 
                                      color=colors[0], 
                                      fmt='o', 
                                      markersize=3.0, 
                                      capsize=2,
                                      label='This work (AW)')
                    
    if theory_fname is not None:
        #loop over redshift bins
         for i in np.arange(nbins_src):
            for j in np.arange(nbins_lens):
                # Reading multipoles
                ell_th = np.loadtxt(os.path.join(theory_fname,f'ell_or_theta_galaxy_sheardensity_cl_e_source_{i}_lens_{j}.txt'))
                # Reading Cells
                Cell_th = np.loadtxt(os.path.join(theory_fname,f'theory_galaxy_sheardensity_cl_e_source_{i}_lens_{j}.txt'))
                # prefactor 
                pref = ell_th * (ell_th + 1) / (2 * np.pi)
                # Compute Dell
                Dell_th = pref * Cell_th * 10 ** 3
                # plot 
                axs[i,j].plot(ell_th, Dell_th, lw=1.2, color='k')

    plt.legend(bbox_to_anchor=(1, 4.25),frameon=False,fontsize=12,ncol=6)
    if save_fig == True:
        plt.savefig('Gammat2pt.png',
            dpi=300,
            bbox_inches='tight')
        plt.savefig('Gammat2pt.pdf',
            dpi=300,
            bbox_inches='tight')
    plt.show()
    plt.close()
    return()

########################################
###   Statistics & goodness-of-fit   ###
########################################
def DataArray(sacc_fname, probe):
    # Read sacc data
    s = sacc.Sacc.load_fits(sacc_fname)
    # Initialize array 
    d = np.array([])
    # HSC specific
    nbins_src = 4
    nbins_lens = 4
    if probe == 'galaxy_shear_cl_ee':
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_src):
                if i >= j:
                    print(f'Data: {i, j}')
                    # read cosmic shear data points
                    ell, Cell = s.get_ell_cl('galaxy_shear_cl_ee', f'source_{i}', f'source_{j}', return_cov=False)
                    # Append to array
                    d = np.append(d, Cell)
    elif probe == 'galaxy_density_cl':
        for i in np.arange(nbins_lens):
                for j in np.arange(nbins_lens):
                    if i == j:
                        # read cosmic shear data points
                        ell, Cell = s.get_ell_cl("galaxy_density_cl", f'lens_{i}', f'lens_{j}', return_cov=False)
                        # append to array
                        d = np.append(d, Cell)
    elif probe == 'galaxy_shearDensity_cl_e':
        #loop over redshift bins
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_lens):
                ell, Cell = s.get_ell_cl('galaxy_shearDensity_cl_e', f'source_{i}', f'lens_{j}', return_cov=False)
                # append to array
                d = np.append(d, Cell)
    return(d)

def TheoryArray(theory_fname,probe):
    # Initialize array 
    t = np.array([])
    # HSC specific
    nbins_src = 4
    nbins_lens = 4
    if probe == 'galaxy_shear_cl_ee':
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_src):
                if i >= j:
                    print(f'Theory: {i,j}')
                    # read cosmic shear data points
                    Cell = np.loadtxt(os.path.join(theory_fname,f'theory_galaxy_shear_cl_ee_source_{i}_source_{j}.txt'))
                    # Append to array
                    t = np.append(t, Cell)
    elif probe == 'galaxy_density_cl':
        for i in np.arange(nbins_lens):
                for j in np.arange(nbins_lens):
                    if i == j:
                        # read cosmic shear data points
                        Cell = np.loadtxt(os.path.join(theory_fname,f'theory_galaxy_density_cl_lens_{i}_lens_{j}.txt'))
                        # append to array
                        t = np.append(t, Cell)
    elif probe == 'galaxy_shearDensity_cl_e':
        #loop over redshift bins
        for i in np.arange(nbins_src):
            for j in np.arange(nbins_lens):
                Cell = np.loadtxt(os.path.join(theory_fname,f'theory_galaxy_sheardensity_cl_e_source_{i}_lens_{j}.txt'))
                # append to array
                t = np.append(t, Cell)
    return(t)

def Covmat(sacc_fname,probe):
    nbins_src = 4
    nbins_lens = 4
    s = sacc.Sacc.load_fits(sacc_fname)
    # Remove not used probes
    s.remove_selection(data_type='galaxy_shearDensity_cl_b')
    s.remove_selection(data_type='galaxy_shear_cl_bb')
    s.remove_selection(data_type='galaxy_shear_cl_be')
    s.remove_selection(data_type='galaxy_shear_cl_eb')
    if probe == 'galaxy_shear_cl_ee':
        s.remove_selection(data_type='galaxy_density_cl')
        s.remove_selection(data_type='galaxy_shearDensity_cl_e')
    elif probe == 'galaxy_density_cl':
        s.remove_selection(data_type='galaxy_shear_cl_ee')
        s.remove_selection(data_type='galaxy_shearDensity_cl_e')
    elif probe == 'galaxy_shearDensity_cl_e':
        s.remove_selection(data_type='galaxy_shear_cl_ee')
        s.remove_selection(data_type='galaxy_density_cl')
    # Extract covariance
    cov = s.covariance.covmat
    return(cov)

def ComputeChisq(sacc_fname, theory_fname, probe, npar):
    print('<< Chisq calculation >>')
    # probe: 'galaxy_shearDensity_cl_e', 'galaxy_density_cl', 'galaxy_shear_cl_ee' or '3x2pt'
    # Extract data array for a specific probe
    d = DataArray(sacc_fname, probe)
    # print(len(d))
    # Extract theory array for a specific probe
    t = TheoryArray(theory_fname, probe)
    # print(len(t))
    # Extract covariance from data
    cov = Covmat(sacc_fname,probe)
    # Covmat inverse
    covinv = np.linalg.inv(cov)
    # Compute chisq
    x = d - t
    xT = x.T
    chisq = np.dot(xT, np.dot(covinv, x))
    print(f'Chisq = {np.round(chisq, 2)}')
    # ndof
    ndata = len(d)
    ndof = ndata - npar
    chisq_ndof = chisq / ndof
    print(f'Chisq / ndof = {np.round(chisq_ndof, 2)}')
    return(chisq, chisq_ndof, ndof)

def ComputeSNR(signal, cov):
    # Inversion of the covariance
    covinv = np.linalg.inv(cov)
    # Compute signal-to-noise ratio
    snr = np.sqrt(np.dot(signal, np.dot(covinv, signal)))
    return(snr)