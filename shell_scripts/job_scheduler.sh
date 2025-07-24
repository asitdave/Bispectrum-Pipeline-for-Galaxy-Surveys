#!/bin/bash

#-------------------------------------- INPUT PARAMETERS --------------------------------------#

JOB_LOG_DIR=" "                              # Directory to store job logs

PARTITION_NAME=                              # Partition (queue) name
TASKS_PER_CORE=2                             # Number of tasks per core
BEGIN=now                                    # Job start time
MAIL_TYPE=ALL                                # notifications for job done & fail
MAIL_USER=                                   # email address for notifications
HINT=compute_bound                           # Job is compute bound
EXCLUDE_NODE=NONE                            # Exclude specific nodes
PROFILE=ALL                                  # Plugin to profile memory usage (Change to ALL to collect all data types)

J1_NTASKS=1                                  # Number of tasks
J1_MEM=150G                                  # Job memory request
J1_CPUS_PER_TASK=8                           # Number of CPU cores per task

J2_NTASKS=1                                  # Number of tasks
J2_MEM=400G                                  # Job memory request (default: 400G)
J2_CPUS_PER_TASK=48                          # Number of CPU cores per task

J3_NTASKS=1                                  # Number of tasks
J3_MEM=100G                                  # Job memory request
J3_CPUS_PER_TASK=8                           # Number of CPU cores per task

#-------------------------------------- DEFINE VARIABLES --------------------------------------#

# Define the array of arguments for job 1
declare -a JOB1_ARGS=("target" "oiii" "siii" "noise") # Arguments for the script

export JOB1_ARGS_STRING="${JOB1_ARGS[*]}"
export REALIZATION=0001
export ZRANGE="0.9-1.1"

# Set the base directory to the directory containing this shell script
BASE_DIR="$(dirname "$(realpath "$0")")"

#-------------------------------------- JOB SETUP --------------------------------------#

JOB1_NAME=Field_computation
JOB1_SCRIPT="$BASE_DIR/../computation_files/compute_fields.py"

JOB2_NAME=bi_terms
JOB2_SCRIPT="$BASE_DIR/../computation_files/compute_bisp_terms.py"

JOB3_NAME=power_terms
JOB3_SCRIPT="$BASE_DIR/../computation_files/compute_ps_terms.py"

#-------------------------------------- JOB EXECUTION --------------------------------------#

# Start timing
SECONDS=0

# Submit the first job (compute fields)
JOB_ID1=$(sbatch --job-name=$JOB1_NAME \
                 --account=$ACCOUNT_NAME \
                 --array=0-$((${#JOB1_ARGS[@]}-1)) \
                 --mem=$J1_MEM \
                 --ntasks=$J1_NTASKS \
                 --cpus-per-task=$J1_CPUS_PER_TASK \
                 --hint=compute_bound \
                 --output=$JOB_LOG_DIR/$JOB1_NAME.%A.%N.out \
                 --error=$JOB_LOG_DIR/$JOB1_NAME.%A.%N.err \
                 --begin=$BEGIN \
                 --mail-type=$MAIL_TYPE \
                 --mail-user=$MAIL_USER \
                 --partition=$J1_PARTITION \
                 --exclude=$EXCLUDE_NODE \
                 runpyscript.sh $JOB1_SCRIPT \
                 | awk '{print $4}')   # Extract job ID

#---------------------------------------------------------------------------------------------#

# Submit the second job (bispectrum terms) after the first one completes successfully
JOB_ID2=$(sbatch --job-name=$JOB2_NAME \
                 --account=$ACCOUNT_NAME \
                 --mem=$J2_MEM \
                 --ntasks=$J2_NTASKS \
                 --cpus-per-task=$J2_CPUS_PER_TASK \
                 --hint=compute_bound \
                 --output=$JOB_LOG_DIR/$JOB2_NAME.%A.%N.out \
                 --error=$JOB_LOG_DIR/$JOB2_NAME.%A.%N.err \
                 --mail-type=$MAIL_TYPE \
                 --mail-user=$MAIL_USER \
                 --partition=$J2_PARTITION \
                 --exclude=$EXCLUDE_NODE \
                 --dependency=afterok:$JOB_ID1 \
                 runpyscript.sh $JOB2_SCRIPT \
                 | awk '{print $4}')   # Extract job ID

# Submit the third job (power spectrum terms) after the first one completes successfully
JOB_ID3=$(sbatch --job-name=$JOB3_NAME \
                 --account=$ACCOUNT_NAME \
                 --mem=$J3_MEM \
                 --ntasks=$J3_NTASKS \
                 --cpus-per-task=$J3_CPUS_PER_TASK \
                 --hint=compute_bound \
                 --output=$JOB_LOG_DIR/$JOB3_NAME.%A.%N.out \
                 --error=$JOB_LOG_DIR/$JOB3_NAME.%A.%N.err \
                 --mail-type=$MAIL_TYPE \
                 --mail-user=$MAIL_USER \
                 --partition=$J3_PARTITION \
                 --exclude=$EXCLUDE_NODE \
                 --dependency=afterok:$JOB_ID1 \
                 runpyscript.sh $JOB3_SCRIPT \
                 | awk '{print $4}')   # Extract job ID


# Submit the fourth job (power spectrum terms) after the first one completes successfully
JOB_ID4=$(sbatch --job-name=$JOB4_NAME \
                 --account=$ACCOUNT_NAME \
                 --mem=$J4_MEM \
                 --ntasks=$J4_NTASKS \
                 --cpus-per-task=$J4_CPUS_PER_TASK \
                 --hint=compute_bound \
                 --output=$JOB_LOG_DIR/$JOB4_NAME.%A.%N.out \
                 --error=$JOB_LOG_DIR/$JOB4_NAME.%A.%N.err \
                 --mail-type=$MAIL_TYPE \
                 --mail-user=$MAIL_USER \
                 --partition=$J4_PARTITION \
                 --exclude=$EXCLUDE_NODE \
                 --dependency=afterok:$JOB_ID1 \
                 runpyscript.sh $JOB4_SCRIPT \
                 | awk '{print $4}')   # Extract job ID

#-------------------------------------- PRINT JOB DETAILS --------------------------------------#

echo "--------------------------------------"
echo "Starting JOB 1 at $(date)"
echo "JOB ID: $JOB_ID1"
echo "Running on partition: $PARTITION_NAME"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $J1_NTASKS processors."
echo "Running on CPUs per task: $J1_CPUS_PER_TASK"
echo "Job 1 name: $JOB1_NAME"
echo "Job 1 script: $JOB1_SCRIPT"
echo "Job 1 arguments: ${JOB1_ARGS[@]}"
echo "--------------------------------------"

echo "--------------------------------------"
echo "Starting JOB 2 at $(date)"
echo "JOB ID: $JOB_ID2"
echo "Running on partition: $PARTITION_NAME"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $J2_NTASKS processors."
echo "Running on CPUs per task: $J2_CPUS_PER_TASK"
echo "Job 2 name: $JOB2_NAME"
echo "Job 2 script: $JOB2_SCRIPT"
echo "--------------------------------------"

echo "--------------------------------------"
echo "Starting JOB 3 at $(date)"
echo "JOB ID: $JOB_ID3"
echo "Running on partition: $PARTITION_NAME"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $J3_NTASKS processors."
echo "Running on CPUs per task: $J3_CPUS_PER_TASK"
echo "Job 2 name: $JOB3_NAME"
echo "Job 2 script: $JOB3_SCRIPT"
echo "--------------------------------------"

echo "--------------------------------------"
echo "Starting JOB 4 at $(date)"
echo "JOB ID: $JOB_ID4"
echo "Running on partition: $PARTITION_NAME"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $J4_NTASKS processors."
echo "Running on CPUs per task: $J4_CPUS_PER_TASK"
echo "Job 2 name: $JOB4_NAME"
echo "Job 2 script: $JOB4_SCRIPT"
echo "--------------------------------------"

#-------------------------------------- CHECK JOB STATUS --------------------------------------#

check_job_status() {
    local job_id=$1
    while true; do
        job_status=$(squeue --job $job_id --noheader --format="%T")
        
        if [ -z "$job_status" ]; then
            echo "Job $job_id has completed."
            break
        else
            echo "Job $job_id is still running or in state: $job_status"
            sleep 180  # Wait for some seconds before checking again
        fi
    done
}

# Check the status of job 2
check_job_status $JOB_ID2
check_job_status $JOB_ID3

#-------------------------------------- PRINT THE JOB IDs AND ELAPSED TIME -------------------#

# Print the job IDs
echo "All jobs have been submitted:"
echo "Job 1 ID: $JOB_ID1"
echo "Job 2 ID: $JOB_ID2"

# Convert to minutes and seconds
minutes=$((SECONDS / 60))
hours=$((minutes / 60))
minutes=$((minutes % 60)) 
seconds=$((SECONDS % 60))

echo "Total time taken to complete the jobs: ${hours} hours ${minutes} minutes and ${seconds} seconds."