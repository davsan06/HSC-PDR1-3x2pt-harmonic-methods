# HSC-PDR1-3x2pt-harmonic-methods

## Data

### Catalogs
Source and lens catalogs live in NERSC. We consider 5 different patches of the sky: GAMA09H, GAMA15H, VVDS, WIDE12H and XMM. Individual field catalogs can be found under `/global/cfs/projectdirs/lsst/groups/LSS/HSC_reanalysis/data_javi/2023_reanalysis`, lens catalogs are named `photometry_lenscatalog_hsc_{FIELD}_nonmetacal_pdr1.h5` and source catalogs are `shear_sourcecatalog_hsc_{FIELDS}_nonmetacal_05_22.h5`. Stacked catalogs with all the info from the different regions of the sky can also be found with the names `photometry_lenscatalog_hsc_ALL_nonmetacal_pdr1_11_06.h5` and `shear_sourcecatalog_hsc_ALL_nonmetacal_11_06.h5`
### Mask
Masks used for the galaxy clustering measurements are saved here `data/mask/star_snr10_imag245/mask_star_snr10_imag245_nside2048_{FIELD}.hs`. This masks are built considering: (i) SNR(i-band) ~ 10, (ii) mag_i > 24.5 and (iii) bright-star mask. See the paper for more details.

### TXPipe configuration files
In NERSC you can find the ini-files in yaml format and the derived data products from tomographic catalogs to the final data vector or summary statistic. Path: `/global/cfs/projectdirs/lsst/groups/LSS/HSC_reanalysis/txpipe`

### Harmonic space
* TXPipe (This work)
- Fiducial data vector ´/data/harmonic/txpipe/source_s16a_lens_dr1/all-fields/dndz/summary_statistics_fourier_all_SourcesS16A_LensesDR1_pz_mc_eab_HikageShearSC_DESC_GCandGGL_SC.sacc´

* Nicola et al.

* Hikage et al. (HSC official analysis)
- Raw txt (as released by HSC) ´/data/harmonic/hikage/raw/band_powers.dat´
- Sacc ´/data/harmonic/hikage/sacc/summary_statistics_fourier_Hik_shear_Cells_Hik_covmat_Hik_dndz.sacc´

### Configuration space
* TXPipe - Longley-Philips and Chang paper

## Scripts
* HSCMeasurementUtils.py 

## Analysis

## MCMC

# Acknowledgments

![LSST Corporation logo](https://noirlab.edu/public/media/archives/logos/screen/logo052.jpg)

We are grateful for the financial support provided by the LSST Corporation for their support through the 2021-02 Enabling Science Program. This funding facilitated the successful execution of our research project and provided DSC the opportunity to visit JS at Fermilab facility. The collaborative environment at Fermilab and the insightful interactions with researchers at the Kavli Institute for Cosmological Physics (KIPC) at the University of Chicago enriched our work and expanded our perspectives. The combined support from the LSST Corporation, Fermilab, and KIPC UChicago has been instrumental in the accomplishments presented in this paper.
