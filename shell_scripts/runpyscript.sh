#!/bin/bash

# Get the Python script to run as an argument
PYTHON_SCRIPT=$1

# Shift to remove the first argument (the script name)
shift

# Initialize the Conda environment
conda activate venv-bipolars

# Set the number of threads Numba will use to the number of CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

IFS=' ' read -r -a JOB1_ARGS <<< "$JOB1_ARGS_STRING"  # Split string back into array

# Define file paths for data and random catalogs based on species
declare -A DATA_PATHS=(
    ["target"]="path_to_data_catalog_for_target_species.fits"
    ["oiii"]="path_to_data_catalog_for_oiii_species.fits"
    ["siii"]="path_to_data_catalog_for_siii_species.fits"
    ["noise"]="path_to_data_catalog_for_noise_species.fits"
)

declare -A RANDOM_PATHS=(
    ["target"]="path_to_random_catalog_for_target_species.fits"
    ["oiii"]="path_to_random_catalog_for_oiii_species.fits"
    ["siii"]="path_to_random_catalog_for_siii_species.fits"
    ["noise"]="path_to_random_catalog_for_noise_species.fits"
)

# Check if SLURM_ARRAY_TASK_ID is set and if any arguments are passed
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Dynamically get the species argument based on SLURM_ARRAY_TASK_ID
    SPECIES_ARG=${JOB1_ARGS[$SLURM_ARRAY_TASK_ID]}
    
    # Lookup paths for the given species
    DATA_CATALOG_PATH=${DATA_PATHS[$SPECIES_ARG]}
    RANDOM_CATALOG_PATH=${RANDOM_PATHS[$SPECIES_ARG]}
    
    echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    echo "Species argument: $SPECIES_ARG"
    echo "Data catalog path: $DATA_CATALOG_PATH"
    echo "Random catalog path: $RANDOM_CATALOG_PATH"
    
    # Execute the Python script with the species and catalog paths arguments
    python3 $PYTHON_SCRIPT --species="$SPECIES_ARG" --data_catalog_path="$DATA_CATALOG_PATH" --randoms_catalog_path="$RANDOM_CATALOG_PATH"
else
    # If there are no arguments to pass, run the script without arguments
    python3 $PYTHON_SCRIPT "$@"
fi

# Deactivate the Conda environment
conda deactivate
