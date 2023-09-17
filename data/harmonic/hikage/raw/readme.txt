1) data vector (filename: band_powers.dat)
 These provide band powers of tomographic lensing spectra l(l+1)C_l/2pi. We adopt 4-bin tomographic analysis and then the number of lensing spectra including auto and cross spectra becomes 10 in total. Each spectrum has 6 data points.

2) error of the data vector (filename: band_errors.dat)
 This provides the 1sigma error of the above data vector.

3) covariance of tomographic lensing spectra (filename: cov_powers.dat)
 Covariance matrix analytically estimated in the best-fit cosmology (the details are written in the Appendix 2 of our paper). The side-length of the matrix is 60 = 6 multipoles times 10 tomographic spectra.

3) equally weighted posterior samples generated from multinest samplers in different setups
  a) Lambda CDM model in fiducial setup (w/o neutrino mass)
    filename: HSC_Y1_LCDM_post_fid.txt
  b) same as a) but for neutrino mass fixed to be 0.06eV
    filename: HSC_Y1_LCDM_post_mnu0.06eV.txt
  c) same as a) but dark energy equation-of-state parameter w varied (wCDM model)
    filename: HSC_Y1_wCDM_post.txt

For questions or if you request other products, contact Chiaki Hikage (chiaki.hikage@ipmu.jp).

When using the above data, please cite the paper by 
 Hikage et al. 2019, PASJ, 71, 43 (arXiv:1809.09148).  

We use the HSC Y1 shear catalog developed by 
 Mandelbaum et al. 2018, PASJ, 70, S25
 Oguri et al. 2018., PASJ, 70, S26
 Mandelbaum et al. 2018, MNRAS, 481, 3170 

The HSC data is processed by 
 Miyazaki et al. 2018, PASJ, 70, S1
 Komiyama et al. 2018, PASJ, 70, S2
 Furusawa et al. 2018, PASJ, 70, S3
 Kawanomoto et al. 2018, PASJ, 70, 66 
and the photometric information processed by 
 Tanaka et al. 2018. PASJ, 70, S9
