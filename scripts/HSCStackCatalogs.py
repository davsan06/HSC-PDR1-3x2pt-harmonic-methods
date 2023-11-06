# Credits David Sanchez
# Date: 06/11/2023

# This script stacks the catalogs of the 5 (no Hectomap) HSC fields into a single catalog
import os
import numpy as np
import h5py

# Path to the catalogs
path = '/global/cfs/projectdirs/lsst/groups/LSS/HSC_reanalysis/data_javi/2023_reanalysis'
fname_source = 'shear_sourcecatalog_hsc_ALL_nonmetacal_11_06.h5'
fname_lens = 'photometry_lenscatalog_hsc_ALL_nonmetacal_pdr1_11_06.h5'

""" # Read fname_source
if os.path.isfile(os.path.join(path,fname_source)):
    print('>> File', fname_source, 'exists')
    print('>> Reading file...')
    f = h5py.File(os.path.join(path,fname_source), 'r')
    # Extract the data from f['shear']
    data = f['shear']
    # Extract the keys of the data
    keys_ref = list(data.keys())
    print('>> Keys of the catalog are:', keys_ref)
    # Iterate over the keys and print the shape of the data
    for key in keys_ref:
        print('>> Shape of the data for key', key, 'is:', data[key].shape)
    # Close the file
    f.close() """

# Source catalogs
list_fields = ['GAMA09H', 'GAMA15H', 'VVDS', 'WIDE12H', 'XMM']

###############################
###   HSC SOURCE CATALOGS   ###
###############################

print('###############################')
print('###   HSC SOURCE CATALOGS   ###')
print('###############################')

# Loop over the fields and stack the catalogs
for field in list_fields:
    print('>> Stacking SOURCE catalogs for field:', field)
    # Initialize the filename of the catalog
    fname = os.path.join(path,f'shear_sourcecatalog_hsc_{field.upper()}_nonmetacal_05_22.h5')
    # Load h5 catalog
    f = h5py.File(fname, 'r')
    # Extract the data from f['shear']
    data = f['shear']
    if field == list_fields[0]:
        # Extract the keys of the data
        keys_ref = list(data.keys())
    # Extract the keys of the data and check they are the same as in the first catalog
    keys = list(data.keys())
    if keys != keys_ref:
        print('>> ERROR: keys of the catalog are not the same as in the first catalog')
        print('>> Exiting...')
        exit()
    else:
        print('>> Keys of the catalog are the same as in the first catalog')
    # Read all the columns and generate and ndarray appending one column at a time
    for col in data.keys():
        # print(col)
        if col == list(data.keys())[0]:
            data_stack = np.array(data[col])
        else:
            # In the rest of the cases, add a new column to the stack
            data_stack = np.vstack((data_stack, np.array(data[col])))
    # Transpose the array
    data_stack = data_stack.T
    print(data_stack.shape)
    # print(data_stack.shape)
    if field == list_fields[0]:
        data_stack_all = data_stack
    else:
        data_stack_all = np.vstack((data_stack_all, data_stack))
    print(data_stack_all.shape)
    # When we get to the last field, save data_stack_all to a new catalog with the same structure
    # as the original catalogs
    if field == list_fields[-1]:
        print('>> Saving stacked catalog...')
        print(fname_source)
        # Create the new catalog
        fname_new = os.path.join(path,fname_source)
        f_new = h5py.File(fname_new, 'w')
        # Create the group 'shear'
        g = f_new.create_group('shear')
        # Create the datasets
        for col in keys:
            g.create_dataset(col, data=data_stack_all[:,keys.index(col)])
        # Close the file
        f_new.close()
        print('>> Done!')

###############################
###   HSC LENS CATALOGS   ###
###############################

print('#############################')
print('###   HSC LENS CATALOGS   ###')
print('#############################')

# Loop over the fields and stack the catalogs
for field in list_fields:
    print('>> Stacking LENS catalogs for field:', field)
    # Initialize the filename of the catalog
    fname = os.path.join(path,f'photometry_lenscatalog_hsc_{field.upper()}_nonmetacal_pdr1.h5')
    # Load h5 catalog
    f = h5py.File(fname, 'r')
    # print(f.keys())
    # Extract the data from f['shear']
    data = f['photometry']
    if field == list_fields[0]:
        # Extract the keys of the data
        keys_ref = list(data.keys())
    # Extract the keys of the data and check they are the same as in the first catalog
    keys = list(data.keys())
    if keys != keys_ref:
        print('>> ERROR: keys of the catalog are not the same as in the first catalog')
        print('>> Exiting...')
        exit()
    else:
        print('>> Keys of the catalog are the same as in the first catalog')
    # Read all the columns and generate and ndarray appending one column at a time
    for col in data.keys():
        # print(col)
        if col == list(data.keys())[0]:
            data_stack = np.array(data[col])
        else:
            # In the rest of the cases, add a new column to the stack
            data_stack = np.vstack((data_stack, np.array(data[col])))
    # Transpose the array
    data_stack = data_stack.T
    print(data_stack.shape)
    # print(data_stack.shape)
    if field == list_fields[0]:
        data_stack_all = data_stack
    else:
        data_stack_all = np.vstack((data_stack_all, data_stack))
    print(data_stack_all.shape)
    # When we get to the last field, save data_stack_all to a new catalog with the same structure
    # as the original catalogs
    if field == list_fields[-1]:
        print('>> Saving stacked catalog...')
        print(fname_lens)
        # Create the new catalog
        fname_new = os.path.join(path,fname_lens)
        f_new = h5py.File(fname_new, 'w')
        # Create the group 'shear'
        g = f_new.create_group('photometry')
        # Create the datasets
        for col in keys:
            g.create_dataset(col, data=data_stack_all[:,keys.index(col)])
        # Close the file
        f_new.close()
        print('>> Done!')
    

