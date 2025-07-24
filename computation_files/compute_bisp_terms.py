"""
BIspectrum and Power spectrum Optimization for LArge Redshift Surveys (BIPOLARS)
-------------------------------------------------
Author: Asit Dave
Date: 03-10-2024

Description:
------------
compute_bisp_terms.py is a part of BIPOLARS pipeline that computes the correlation terms of
bispectrum multipoles for Euclid-like galaxy surveys.

- This script auto- and cross-terms for the bispectrum multipoles for given number of species. 
- The script assumes the same naming convection as defined in the <job_scheduler.sh> script for the species involved in the catalog.
  For example, the Fx-file is expected to be named as: F0_x_target.npy, F2_x_target.npy, F0_x_interloper1.npy, etc.
- The script saves the computed cross-terms in a pickle file.
"""

#----------------------------------------------- Import libraries -----------------------------------------------#

from bipolars import BispectrumTerms
from compute_fields import FIELDS_SAVE_DIR, CATALOG_INFO_SAVE_DIR, NUM_TRIANGLES_FILE_PATH, KCONFIG_FILE_PATH, REALIZATION, ZRANGE, fourier_config

import numpy as np
import logging
import os
import time
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)

#----------------------------------------------- Configure the script -----------------------------------------------#

max_workers = int(os.getenv('OMP_MAX_ACTIVE_LEVELS', os.cpu_count()))

# Set the OpenMP environment variable
os.environ['OMP_MAX_ACTIVE_LEVELS'] = str(max_workers)

# Define the path to save the computed cross terms
FILE_SAVE_DIR = "PATH_TO_SAVE_COMPUTED_TERMS" # TODO: Change this path

# Define the name of the pickle file to save the computed cross terms
PICKLE_FILENAME = f"bisp_terms_L{fourier_config['L']}_N{fourier_config['N']}_R{REALIZATION}_z{ZRANGE}.pkl"

# Naming convention for different species
species_names = os.getenv('JOB1_ARGS_STRING').split() if os.getenv('JOB1_ARGS_STRING') else ['target', 'oiii', 'siii', 'noise']
species_names = [name.lower() for name in species_names]

catalog_properties_files = [os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_{species}_R{REALIZATION}_z{ZRANGE}.pkl") for species in species_names]

# Load catalog properties (I22)
def load_catalog_properties(filepath):
    with open(filepath, 'rb') as f:
        catalog_properties = pickle.load(f)
    return catalog_properties

I33_dict = {species_names[i]: load_catalog_properties(file)['I33.randoms'] for i, file in enumerate(catalog_properties_files)}
I33_dict['total'] = load_catalog_properties(os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_total_R{REALIZATION}_z{ZRANGE}.pkl"))['I33.randoms']
logging.info(f"I33: {I33_dict}")

#----------------------------------------------- Load Data -----------------------------------------------#
if __name__ == "__main__":

  logging.info(f"Starting computation for power spectrum terms (z: {ZRANGE} and realization: {REALIZATION})...")
  
  start_time = time.time()

  # Load k-configurations and number of triangles per bin
  k_configs = np.load(KCONFIG_FILE_PATH)
  num_triangles_per_bin = np.load(NUM_TRIANGLES_FILE_PATH)

  #----------------------------------------------- Compute Bispectrum Terms -----------------------------------------------#

  # Initialize BispectrumTerms instance
  bisp_terms_calculator = BispectrumTerms(species_names, num_triangles_per_bin, k_configs, I33_dict, [0, 2])

  # Specify directory containing precomputed F_x field files
  Fx_save_dir = FIELDS_SAVE_DIR

  # Load fields
  Fx_files = bisp_terms_calculator.import_fields(Fx_save_dir, workers=max_workers)

  # Create directory to save computed bispectrum terms
  os.makedirs(FILE_SAVE_DIR, exist_ok=True)

  # Specify save path for computed bispectrum terms
  file_save_path = os.path.join(FILE_SAVE_DIR, PICKLE_FILENAME)

  # Perform computation
  all_bispectrum_terms = bisp_terms_calculator.compute(Fx_files, file_save_path, workers=max_workers)

  all_data_dict = {}  
  all_data_dict.update(bisp_terms_calculator.f_c)
  all_data_dict.update(bisp_terms_calculator.prefacts)
  
  with open(os.path.join(FILE_SAVE_DIR, f"bisp_info_L{fourier_config['L']}_N{fourier_config['N']}_R{REALIZATION}_z{ZRANGE}.pkl"), 'wb') as f:
      pickle.dump(all_data_dict, f)

  # Log time taken
  logging.info("Time taken: %s seconds", time.time() - start_time)

#----------------------------------------------- End of Script -----------------------------------------------#