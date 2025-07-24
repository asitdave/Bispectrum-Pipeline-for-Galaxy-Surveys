"""
BIspectrum and Power spectrum Optimization for LArge Redshift Surveys (BIPOLARS)
-------------------------------------------------
Author: Asit Dave
Date: 02-10-2024

Description:
------------
compute_fields.py is a part of the BIPOLARS pipeline that generates F_k and F_x fields along with other necessary computations,
for a given galaxy species from Euclid-like survey catalogs. This script is essential for preparing inputs 
used in bispectrum multipole calculations.

- This script loads galaxy and random catalogs, applies the FKP weighting scheme, and computes F_k and F_x fields 
  in Fourier and real space, respectively. It saves these fields in .npy format, following a defined naming convention.
- The script also saves key catalog properties, k-bins, k-configurations, and precomputed triangle information 
  needed for bispectrum computation.

Notes:
------
- Ensure paths are updated as needed, especially for output and reference file directories.
- The same naming convention as defined in <job_scheduler.sh> is used here for consistency.
"""

from bispigals import FKPCatalogManager, ComputeFields, Bispectrum
from nbodykit.lab import cosmology
from nbodykit import setup_logging

import numpy as np
import numba as nb

import os
import logging
import argparse
import time
import pickle

# Set up logging
setup_logging(log_level='info')

# ------------------------------------------- Configure the script ---------------------------------------------------- #

# Realization number of the catalog
REALIZATION = os.getenv('REALIZATION') if os.getenv('REALIZATION') is not None else "0001"

# Redshift range of the catalog
ZRANGE = os.getenv('ZRANGE') if os.getenv('ZRANGE') is not None else "0.9-1.1" 

# Output path
OUTPUT_CATALOG_INFO_FILE = "CatalogInfo.txt"
FIELDS_SAVE_DIR = "PATH_TO_SAVE_FIELDS" # TODO: Change this path

CATALOG_INFO_SAVE_DIR = "PATH_TO_SAVE_CATALOG_INFO" # TODO: Change this path

# Paths to save the fields
REFERENCE_FILES_DIR = "PATH_TO_SAVE_REFERENCE_FILES" # TODO: Change this path

K_BIN_INDICES_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "k-bin_indices_info.pkl")
KBIN_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "k_bins.npy")
KCONFIG_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "k_configs.npy")
NUM_TRIANGLES_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "num_triangles.npy")
# ----------------------------------------------------------------------------------------------- #

# --- Cosmology ---
cosmo = cosmology.Cosmology(h=0.67, Omega0_cdm=0.27, Omega0_b=0.049)

# --- Configuration for Fourier algorithm ---
N = 450
L = 3000
k_funda = 2 * np.pi / L
k_nyquist = np.pi * N / L

dk = 0.009
kmin =  0 + dk/2
kmax = 0.3

# Configuration for Fourier algorithm
fourier_config = {
    'N': N,  # Number of grid points per axis
    'L': L,  # Box size (Mpc/h)
    'dk': dk,  # k-bin width
    'kmin': kmin,  # Minimum k value
    'kmax': kmax,  # Maximum k value (optional, default Nyquist) 
    'poles': [0, 2],  # List of poles to compute
}

# ----------------------------------------------------------------------------------------------- #
if __name__ == "__main__":

    try:
        max_threads = int(os.getenv('SLURM_CPUS_PER_TASK')) # Number of workers for parallel processing
    except:
        max_threads = os.cpu_count()
    
    nb.set_num_threads(max_threads) # Number of threads for numba functions

    logging.info(f"Using {max_threads} workers for parallel computation.")

    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description="Compute Fx-fields for a given galaxy catalog.")
    parser.add_argument("--species", type=str, help="Name of the species in the catalog for which you want to calculate the Fx fields.")
    parser.add_argument("--data_catalog_path", type=str, help="Path to the data catalog.")
    parser.add_argument("--randoms_catalog_path", type=str, help="Path to the randoms catalog.")

    args = parser.parse_args()

    if args.species:
        SPECIES = args.species.lower()
    else:
        raise ValueError("Please provide the species name in the catalog for which you want to calculate the Fx fields.")
    
    logging.info(f"Computing fields for Species: {SPECIES}")
    logging.info(f"Realization: {REALIZATION}")
    logging.info(f"Redshift range: {ZRANGE}")

    DATA_CATALOG_PATH = args.data_catalog_path
    RANDOMS_CATALOG_PATH = args.randoms_catalog_path

    start_time = time.time()

    logging.info(f"Data catalog path: {DATA_CATALOG_PATH}")
    logging.info(f"Randoms catalog path: {RANDOMS_CATALOG_PATH}")

    # Check if the catalog paths are valid
    if not os.path.exists(DATA_CATALOG_PATH):
        raise FileNotFoundError(f"Data catalog file not found at {DATA_CATALOG_PATH}")
    if not os.path.exists(RANDOMS_CATALOG_PATH):
        raise FileNotFoundError(f"Randoms catalog file not found at {RANDOMS_CATALOG_PATH}")

    # --- Step 0: Create the output directory ---
    os.makedirs(FIELDS_SAVE_DIR, exist_ok=True)
    os.makedirs(REFERENCE_FILES_DIR, exist_ok=True) 
    os.makedirs(CATALOG_INFO_SAVE_DIR, exist_ok=True)

    # --- Step 1: Load Catalog and Write Catalog Info to File ---
    # Create the FKP catalog manager instance
    nz_data_hist = RedshiftHistogram.load(os.path.join(REFERENCE_FILES_DIR, f"nz_zhist", f"nz_z{ZRANGE}_R{REALIZATION}_data.json"))
    nz_randoms_hist = RedshiftHistogram.load(os.path.join(REFERENCE_FILES_DIR, f"nz_zhist", f"nz_z{ZRANGE}_R{REALIZATION}_randoms.json"))

    fkp_manager = FKPCatalogManager(data_catalog_path=DATA_CATALOG_PATH, 
                                    randoms_catalog_path=RANDOMS_CATALOG_PATH, 
                                    cosmo=cosmo,
                                    convert_to_cartesian=True
                                    )
    
    data_catalog, randoms_catalog = fkp_manager.compute_nofz(zhist_data=nz_data_hist, zhist_randoms=nz_randoms_hist)

    # Get catalog info and write it to a file
    catalog_info = fkp_manager.get_catalog_info()
    with open(os.path.join(CATALOG_INFO_SAVE_DIR, f"CatalogInfo_{SPECIES}_R{REALIZATION}_z{ZRANGE}.txt"), 'w') as file:
        file.write('--- Data Catalog Information ---\n')
        file.write(catalog_info['header_info'])

    logging.info(f"Galaxy catalog information written to CatalogInfo_{SPECIES}.txt")

    # --- Step 2: Create FKP Catalog ---
    fkp_catalog = fkp_manager.to_FKPCatalog(P0=2e4)

    # --- Step 3: Compute F_k Field ---
    if os.path.exists(NUM_TRIANGLES_FILE_PATH):
        fields = ComputeFields(fkp_catalog=fkp_catalog, config=fourier_config, Fk_field=True, Fx_field=True, num_triangles_path=NUM_TRIANGLES_FILE_PATH)  
    else:
        fields = ComputeFields(fkp_catalog=fkp_catalog, config=fourier_config, Fk_field=True, Fx_field=True)

    # --- Step 4: Save the properties of the fields ---
    CATALOG_PROPERTIES_FILE = os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_{SPECIES}_R{REALIZATION}_z{ZRANGE}.txt")

    logging.info("Saving field properties...")
    with open(CATALOG_PROPERTIES_FILE, "w") as file:
        file.write(repr(fields.properties))
    logging.info(f"Field properties saved successfully.")

    # ----------------------------------------------------------------------------------------------- #

        # --- Step 5: Save Fk fields --- 
    logging.info("Saving F(k) fields...")
    np.save(os.path.join(FIELDS_SAVE_DIR, f"F0_k_{SPECIES}.npy"), fields.F0k_field) 
    logging.info(f"F0k_{SPECIES} field saved successfully.")
    if 2 in fourier_config['poles']:
        np.save(os.path.join(FIELDS_SAVE_DIR, f"F2_k_{SPECIES}.npy"), fields.F2k_field)
        logging.info(f"F2k_{SPECIES} field saved successfully.")

    # --- Step 6: Save Fx fields --- 
    logging.info("Saving F(x) fields...")
    np.save(os.path.join(FIELDS_SAVE_DIR, f"F0_x_{SPECIES}.npy"), fields.F0x_field)
    logging.info(f"F0x_{SPECIES} field saved successfully.")
    if 2 in fourier_config['poles']:
        np.save(os.path.join(FIELDS_SAVE_DIR, f"F2_x_{SPECIES}.npy"), fields.F2x_field) 
        logging.info(f"F2x_{SPECIES} field saved successfully.")

    # --- Step 7: Save k-bins & k-configurations --- 
    if not os.path.exists(KBIN_FILE_PATH):
        np.save(KBIN_FILE_PATH, fields.k_bins)
        logging.info("k_bins saved successfully.")
    else:
        logging.info("k_bins already exists.")
    
    if not os.path.exists(KCONFIG_FILE_PATH):
        np.save(KCONFIG_FILE_PATH, fields.k_configs)
        logging.info("k_configs saved successfully.")
    else:
        logging.info("k_configs already exists.")

    # Save k_bin_indices to a pickle file
    if not os.path.exists(K_BIN_INDICES_FILE_PATH):
        with open(K_BIN_INDICES_FILE_PATH, 'wb') as file:
            pickle.dump(fields.k_bin_info, file)
        logging.info("k-bin_indices_info.pkl saved successfully.")
    else:
        logging.info("k-bin_indices_info.pkl already exists.")

    # --- Step 8: Save Number of triangles per bin ---
    if not os.path.exists(NUM_TRIANGLES_FILE_PATH):
        np.save(NUM_TRIANGLES_FILE_PATH, fields.num_triangles())
        logging.info("num_triangles saved successfully.")
    else:
        logging.info("num_triangles already exists.")
    
    # # --- Step 9: Compute & Save Bispectrum ---
    # bispec = Bispectrum(Fields=fields)
    # bispec.compute()
    # BISPEC_SAVE_PATH = "/vol/calvin/data/adave/Thesis/bispectrum"
    # os.makedirs(BISPEC_SAVE_PATH, exist_ok=True)
    # bispec.save(save_path=BISPEC_SAVE_PATH, species=SPECIES)

    logging.info(f"Total time taken: {time.time() - start_time} seconds.")