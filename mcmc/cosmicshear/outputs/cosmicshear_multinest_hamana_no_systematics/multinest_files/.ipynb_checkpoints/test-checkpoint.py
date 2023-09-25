import nestcheck.data_processing

base_dir = '/pscratch/sd/d/davidsan/new_firecrown/firecrown/examples/hsc_3x2pt/cosmicshear/output/cosmicshear_multinest_hamana_no_systematics/multinest_files'  # directory containing run (PolyChord's 'base_dir' setting)
file_root = 'mn_outfile_cosmicshear_hamana_no_systematics_'  # output files' name root (PolyChord's 'file_root' setting)
run = nestcheck.data_processing.process_multinest_run(file_root, base_dir)