"""
BIspectrum and Power spectrum Optimization for LArge Redshift Surveys (BIPOLARS)
-------------------------------------------------
Author: Asit Dave
Date: 04-10-2024

Description:
------------
compute_ps_terms.py is part of the BIPOLARS pipeline that calculates the power spectrum multipole terms 
for galaxy survey data, specifically for an Euclid-like survey catalog.

- This script computes the power spectrum multipoles for auto- and cross-terms between specified galaxy species. It 
  calculates terms up to the specified multipole order and includes corrections for randoms based on 
  catalog properties.
- The computed power spectrum terms are saved in a pickle file, following a structured naming convention.
  
Notes:
------
- The script expects k-bin indices and F(k) field files as inputs, which are used in computing the 
  power spectrum multipole terms.
- Output paths should be updated before running the script.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from bipolars import PowerSpectrumTerms
from compute_fields import FIELDS_SAVE_DIR, CATALOG_INFO_SAVE_DIR, K_BIN_INDICES_FILE_PATH, REALIZATION, fourier_config, ZRANGE

import logging
import os
import time
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)

#----------------------------------------------- Configure the script -----------------------------------------------#

max_workers = int(os.getenv('SLURM_CPUS_PER_TASK')) if os.getenv('SLURM_CPUS_PER_TASK') is not None else 4 # Number of workers for parallel processing

# Define the path to save the computed cross terms
FILE_SAVE_DIR = "PATH_TO_SAVE_COMPUTED_TERMS" # TODO: Change this path

# Define the name of the pickle file to save the computed cross terms
PICKLE_FILENAME = f"ps_terms_L{fourier_config['L']}_N{fourier_config['N']}_R{REALIZATION}_z{ZRANGE}.pkl"

# Naming convention for different species
species_names = os.getenv('JOB1_ARGS_STRING').split() if os.getenv('JOB1_ARGS_STRING') else ['target', 'oiii', 'siii', 'noise']
species_names = [name.lower() for name in species_names]

catalog_properties_files = [os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_{species}_R{REALIZATION}_z{ZRANGE}.pkl") for species in species_names]

# Load catalog properties (I22)
def load_catalog_properties(filepath):
    with open(filepath, 'rb') as f:
        catalog_properties = pickle.load(f)
    return catalog_properties

I22s = {species_names[i]: load_catalog_properties(file)['I22.randoms'] for i, file in enumerate(catalog_properties_files)}
I22s['total'] = load_catalog_properties(os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_total_R{REALIZATION}_z{ZRANGE}.pkl"))['I22.randoms']
logging.info(f"I22: {I22s}")

N0s = {species_names[i]: load_catalog_properties(file)['shotnoise2pt'] for i, file in enumerate(catalog_properties_files)}
logging.info(f"N0s: {N0s}")

#-------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    logging.info(f"Starting computation for power spectrum terms (z: {ZRANGE} and realization: {REALIZATION})...")

    start_time = time.time()

    # Load k-bin indices pkl file
    logging.info("Loading k-bin indices...")
    with open(K_BIN_INDICES_FILE_PATH, 'rb') as f:
        k_bin_indices = pickle.load(f)

    logging.info(f"Loaded k-bin indices.")

    #----------------------------------------------- Compute Power Spectrum Terms -----------------------------------------------#

    # Initialize BispectrumTerms instance
    ps_terms_calculator = PowerSpectrumTerms(species_names, k_bin_indices, poles=[0, 2], I22=I22s, N0=N0s)

    # Specify directory containing precomputed F_x field files
    Fk_save_dir = FIELDS_SAVE_DIR

    # Load fields
    logging.info("Loading F(k) fields...")
    Fk_files = ps_terms_calculator.import_fields(Fk_save_dir, workers=max_workers)
    logging.info(f"F(k) fields loaded successfully.")

    # Specify save path for computed bispectrum terms
    file_save_path = os.path.join(FILE_SAVE_DIR, PICKLE_FILENAME)

    # Create directory to save computed bispectrum terms
    os.makedirs(FILE_SAVE_DIR, exist_ok=True)

    # Perform computation
    logging.info("Computing power spectrum terms...")
    all_power_terms = ps_terms_calculator.compute(Fk_files, file_save_path)
    logging.info(f"Power spectrum terms computed successfully.")

    all_data_dict = {}  
    all_data_dict.update(ps_terms_calculator.f_c)
    all_data_dict.update(ps_terms_calculator.prefacts)

    with open(os.path.join(FILE_SAVE_DIR, f"ps_info_L{fourier_config['L']}_N{fourier_config['N']}_R{REALIZATION}_z{ZRANGE}.pkl"), "wb") as f:
        pickle.dump(all_data_dict, f)

    # Log time taken
    logging.info("Time taken: %s seconds", time.time() - start_time)

#----------------------------------------------- End of Script -----------------------------------------------#