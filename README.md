# Bispectrum-Pipeline-for-Galaxy-Surveys (BisPiGalS)

## Overview
BisPiGalS (Bispectrum Pipeline for Galaxy Surveys) is a Python-based pipeline designed for analyzing galaxy surveys and computing higher-order statistics like the bispectrum and power spectrum multipoles for Euclid-like galaxy surveys. This suite of scripts is optimized for higher-order correlation terms in galaxy catalogs with multiple species, making it an essential tool for cosmological research.

## Pipeline components
The pipeline is divided into several scripts, each performing a specific role in computing the higher-order statistics. The following is an overview of the core scripts included in BisPiGalS:

1. `compute_fields.py`: This script essentially reads through the catalog and generates the F(k) and F(x) fields for each catalog, which is required for the computation of power spectrum and bispectrum. This also saves important properties of the catalog and computations within for user reference.

2. `compute_bisp_terms`: This script optimized to calculate the correlation terms for all the species in the catalog. It stores the result as a dictionary in a pickle file for further analysis.

3. `compute_ps_terms`: This script computes the power spectrum cross and auto-correlation terms for each species in the galaxy survey catalog. Each term is further scaled by their corresponding species fraction. Results are further stored in a pickle file as a dictionary.

## Features
- **Cross-Species Correlation**: Easily compute bispectrum terms for multiple galaxy species (e.g., target, oiii, siii, noise).
- **Parallelized Processing**: Leverages multi-threading and multiprocessing to optimize for high-performance computing environments.
- **Integration with nbodykit**: Utilizes the nbodykit library for cosmology-related functions, enabling accurate and efficient data transformations.
- **Modular Structure**: Each script is standalone, allowing specific components of the pipeline to be run independently based on need.


## Usage
The pipeline is configured to run on an HPC (High-Performance Computing) environment with SLURM. The main control script, `job_scheduler.sh`, schedules and runs the pipeline’s stages.
1. Set Environment Variables:
The scripts use environment variables to specify realization numbers, redshift ranges, and other parameters. Configure these in `job_scheduler.sh`.

2. Run Pipeline Jobs:
Each stage of the pipeline is run through `job_scheduler.sh`, which submits jobs to the HPC environment. For example:
```bash
srun job_scheduler.sh
```

## Repository Structure
```
BisPiGalS/
├── bispigals.py
├── computation_files/                   
│   ├── compute_fields.py             
│   ├── compute_bisp_terms.py                               
│   └── compute_ps_terms.py                   
├── shell_scripts/
│   ├── job_scheduler.sh             
│   ├── runpyscript.sh 
├── README.md                                                 
└── venv-bispigals.yml                    
```

Feel free to fork this repository, explore the code, and contribute by submitting issues or pull requests. Suggestions are always welcome!
