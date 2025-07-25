#!/bin/bash
#SBATCH --account=nct_328
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name="run_eval"
#SBATCH --output=logs/stdout_eval.txt
#SBATCH --error=logs/stderr_eval.txt
#SBATCH --qos=acc_interactive

module purge
module load anaconda/2023.07

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate dl4

# Load parameters from the last training run
CONFIG_FILE="configs/last_run_params.env"
if [ -f "${CONFIG_FILE}" ]; then
    echo "Loading parameters from ${CONFIG_FILE}"
    source "${CONFIG_FILE}"
else
    echo "Error: Config file ${CONFIG_FILE} not found." 
    echo "Please run run_train.sh first or manually create the config file."
    exit 1
fi

# Check if necessary variables are set
if [ -z "${OOD_DATASET}" ] || [ -z "${AE_BOTTLENECK}" ] || [ -z "${CLASSIFIER_DIR}" ] || [ -z "${AUTOENCODER_DIR}" ]; then
    echo "Error: One or more necessary parameters (OOD_DATASET, AE_BOTTLENECK, CLASSIFIER_DIR, AUTOENCODER_DIR) are missing from ${CONFIG_FILE}."
    exit 1
fi

echo "Using OOD: ${OOD_DATASET}"
echo "Using AE Bottleneck: ${AE_BOTTLENECK}"
echo "Using Classifier from: ${CLASSIFIER_DIR}"
echo "Using Autoencoder from: ${AUTOENCODER_DIR}"

python src/evaluate.py --ood "${OOD_DATASET}" \
                       --classifier_experiment_dir "${CLASSIFIER_DIR}" \
                       --autoencoder_experiment_dir "${AUTOENCODER_DIR}" \
                       --ae_bottleneck_dim "${AE_BOTTLENECK}"

echo "Evaluation complete."