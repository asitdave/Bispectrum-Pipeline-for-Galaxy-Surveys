"""
This script is used to compute the shotnoise for the 3-point statistics for the given number of species in the catalog.
This script assumes that you have the FW fields saved, as shown in compute_fields.py script.
"""

import itertools
import numpy as np
import os
import pickle
from scipy.special import legendre

#----------------------------------------------- Configure the script -----------------------------------------------#

# You can create your own species combinations here
# For example, if you have species 'target', 'oiii', 'siii', 'noise', you can create combinations like:
# sorted(itertools.combinations(['t', 'o', 's', 'n'], r=2))

combs = ['o_t', 's_t', 'n_t', 'o_s', 'n_o', 'n_s'] # sorted(itertools.combinations(['t', 'o', 's', 'n'], r=2))

def spherical_average(field: np.ndarray, k_bin_info: dict) -> np.ndarray:
    """
    Computes the spherical average of a field within each k-bin.

    This method averages the input field over specified k-bins defined by the `k_bin_indices` attribute.

    Parameters:
    -----------
    field : np.ndarray
        Input field to average over the defined k-bins.

    Returns:
    --------
    np.ndarray
        Array of spherical averages, one for each k-bin.
    """
    spherical_averages = []
    for k_bin, indices in k_bin_info.items():
        if (len(indices[0]) > 0): 
            # Compute the average of the values for specific k-bin
            spherical_averages.append(np.mean(field[indices]))
        else:
            # If there are no indices in this bin, set the average to NaN
            spherical_averages.append(np.nan)
    return np.array(spherical_averages).real

# Realization number of the catalog
REALIZATION = os.getenv('REALIZATION') if os.getenv('REALIZATION') is not None else "0001"
# Redshift range of the catalog
ZRANGE = os.getenv('ZRANGE') if os.getenv('ZRANGE') is not None else "1.1-1.3" 

print(f"REALIZATION: {REALIZATION}")
print(f"ZRANGE: {ZRANGE}")


FIELDS_SAVE_DIR = "PATH_TO_THE_DIR" # TODO: Change this path
REFERENCE_FILES_DIR = "PATH_TO_THE_DIR" # TODO: Change this path
CATALOG_INFO_SAVE_DIR = "PATH_TO_THE_DIR" # TODO: Change this path
K_BIN_INDICES_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "k-bin_indices_info.pkl")
KCONFIG_FILE_PATH = os.path.join(REFERENCE_FILES_DIR, "k_configs.npy")

k_configs = np.load(KCONFIG_FILE_PATH)

species_names = ['target', 'oiii', 'siii', 'noise']
catalog_properties_files = [os.path.join(CATALOG_INFO_SAVE_DIR, f"field_properties_{species}_R{REALIZATION}_z{ZRANGE}.pkl") for species in species_names]

# Load catalog properties (I22)
def load_catalog_properties(filepath):
    with open(filepath, 'rb') as f:
        catalog_properties = pickle.load(f)
    return catalog_properties

I33s = {species_names[i]: load_catalog_properties(file)['I33.randoms'] for i, file in enumerate(catalog_properties_files)}

with open(K_BIN_INDICES_FILE_PATH, 'rb') as f:
    k_bin_info = pickle.load(f)

# Load the fields
F0k = {
    't': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_k_target_z{ZRANGE}.npy")),
    'o': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_k_oiii_z{ZRANGE}.npy")),
    's': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_k_siii_z{ZRANGE}.npy")),
    'n': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_k_noise_z{ZRANGE}.npy"))
}

F2k = {
    't': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_k_target_z{ZRANGE}.npy")),
    'o': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_k_oiii_z{ZRANGE}.npy")),
    's': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_k_siii_z{ZRANGE}.npy")),
    'n': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_k_noise_z{ZRANGE}.npy"))
}

F0w = {
    't': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_w_target_z{ZRANGE}.npy")),
    'o': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_w_oiii_z{ZRANGE}.npy")),
    's': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_w_siii_z{ZRANGE}.npy")),
    'n': np.load(os.path.join(FIELDS_SAVE_DIR, f"F0_w_noise_z{ZRANGE}.npy"))
}

F2w = {
    't': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_w_target_z{ZRANGE}.npy")),
    'o': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_w_oiii_z{ZRANGE}.npy")),
    's': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_w_siii_z{ZRANGE}.npy")),
    'n': np.load(os.path.join(FIELDS_SAVE_DIR, f"F2_w_noise_z{ZRANGE}.npy"))
}

def cosine(k1, k2, k3):
            return (k1**2 + k2**2 - k3**2)/(2*k1*k2)

# Precompute the Legendre polynomials for l=2
cosines_k2 = np.array([cosine(k1, k2, k3) for (k1, k2, k3) in k_configs])
cosines_k3 = np.array([cosine(k1, k3, k2) for (k1, k2, k3) in k_configs])

L_k2= legendre(2)(cosines_k2)
L_k3 = legendre(2)(cosines_k3)

sp_keys = {
    't': 'target',
    'o': 'oiii',
    's': 'siii',
    'n': 'noise'
}

shotnoise_3pt_dict = {}

for i, sp_comb in enumerate(combs):

    sp1, sp2 = sp_comb.split('_')

    print(f"Computing for {sp1}{sp2} ({i+1}/{len(combs)})")

    perms = sorted(set(''.join(tup) for tup in itertools.product(f'{sp1}{sp2}', repeat=3)))
    perms.remove(f'{sp1}{sp1}{sp1}')
    perms.remove(f'{sp2}{sp2}{sp2}')

    print(f"Perms: {perms}")

    print(f"Computing normalization1: {sp_keys[sp1]} {sp_keys[sp1]} {sp_keys[sp2]}")
    print(f"Computing normalization2: {sp_keys[sp2]} {sp_keys[sp2]} {sp_keys[sp1]}")
    
    norm1 = np.power(I33s[sp_keys[sp1]] * I33s[sp_keys[sp1]] * I33s[sp_keys[sp2]], 1/3) # sp1-sp1-sp2
    norm2 = np.power(I33s[sp_keys[sp2]] * I33s[sp_keys[sp2]] * I33s[sp_keys[sp1]], 1/3) # sp2-sp2-sp1

    # Power spectrum of fields for monopole shotnoise
    SN_term_k123_l0_1 = spherical_average(F0k[sp2]*np.conjugate(F0w[sp1]), k_bin_info) / norm1 # sp1-sp1-sp2
    SN_term_k123_l0_2 = spherical_average(F0k[sp1]*np.conjugate(F0w[sp2]), k_bin_info) / norm2 # sp2-sp2-sp1

    # Power spectrum of fields for quadrupole shotnoise (first term)
    SN_term_k1_l2_1 = 5 * spherical_average(F2k[sp2]*np.conjugate(F0w[sp1]), k_bin_info) / norm1 # sp1-sp1-sp2
    SN_term_k1_l2_2 = 5 * spherical_average(F2k[sp1]*np.conjugate(F0w[sp2]), k_bin_info) / norm2 # sp2-sp2-sp1

    # Power spectrum of fields for quadrupole shotnoise (second and third term)
    SN_term_k2_k3_l2_1 = 5 * spherical_average(F0k[sp2]*np.conjugate(F2w[sp1]), k_bin_info) / norm1 # sp1-sp1-sp2
    SN_term_k2_k3_l2_2 = 5 * spherical_average(F0k[sp1]*np.conjugate(F2w[sp2]), k_bin_info) / norm2 # sp2-sp2-sp1
    
    for perm in perms:
        species1, species2, species3 = perm

        # Compute the shot noise for 3-point statistics
        SN_3pt_0, SN_3pt_2 = [], []

        for idx, (k1, k2, k3) in enumerate(k_configs):
            if species2==species3:
                if species2==sp1: # Case for sp1-sp1-sp2 
                    # Use 1st Term, first equation
                    SN_3pt_0.append(SN_term_k123_l0_1[k1-1]) # Monpole shot noise
                    SN_3pt_2.append(SN_term_k1_l2_1[k1-1]) # Quadrupole shot noise
                    
                else: # Case for sp1-sp2-sp2
                    # Use 1st Term, second equation
                    SN_3pt_0.append(SN_term_k123_l0_2[k1-1]) # Monpole shot noise
                    SN_3pt_2.append(SN_term_k1_l2_2[k1-1]) # Quadrupole shot noise

            elif species1==species3:
                if species1 == sp1:
                    # Use 2nd term, first equation
                    SN_3pt_0.append(SN_term_k123_l0_1[k2-1])
                    SN_3pt_2.append(L_k3[idx] * SN_term_k2_k3_l2_1[k2-1])
                
                else:
                    # Use 2nd term, second equation
                    SN_3pt_0.append(SN_term_k123_l0_2[k2-1])
                    SN_3pt_2.append(L_k3[idx] * SN_term_k2_k3_l2_2[k2-1])

            elif species1==species2:
                if species1 == sp1: 
                    # Use 3rd term, first equation
                    SN_3pt_0.append(SN_term_k123_l0_1[k3-1]) # Monpole shot noise
                    SN_3pt_2.append(L_k2[idx] * SN_term_k2_k3_l2_1[k3-1]) # Quadrupole shot noise

                else:
                    # Use 3rd term, second equation
                    SN_3pt_0.append(SN_term_k123_l0_2[k3-1]) # Monpole shot noise
                    SN_3pt_2.append(L_k2[idx] * SN_term_k2_k3_l2_2[k3-1]) # Quadrupole shot noise
  

        SN_3pt_0 = np.array(SN_3pt_0)
        SN_3pt_2 = np.array(SN_3pt_2)

        shotnoise_3pt_dict[f'{species1}{species2}{species3}_0'] = SN_3pt_0 
        shotnoise_3pt_dict[f'{species1}{species2}{species3}_2'] = SN_3pt_2 


# Save the shotnoise dictionary
with open(os.path.join(CATALOG_INFO_SAVE_DIR, f"shotnoise_3pt_R{REALIZATION}_z{ZRANGE}.pkl"), 'wb') as f:
    pickle.dump(shotnoise_3pt_dict, f)

print("Shotnoise 3pt dictionary saved successfully. File path: ", os.path.join(CATALOG_INFO_SAVE_DIR, f"shotnoise_3pt_R{REALIZATION}_z{ZRANGE}.pkl"))