#!/bin/bash
#SBATCH --account=nct_328
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name="run_train"
#SBATCH --output=logs/stdout_train.txt
#SBATCH --error=logs/stderr_train.txt
#SBATCH --qos=acc_interactive

module purge
module load anaconda/2023.07

eval "$(conda shell.bash hook)"

conda activate dl4

# Define parameters for this training run
OOD_DATASET="tinyimagenet" # Or svhn, cifar100
AE_BOTTLENECK=64
CLASSIFIER_EPOCHS=130
AUTOENCODER_EPOCHS=100

# 1. Train / fine-tune classifier
echo "Starting classifier training..."
CLASSIFIER_OUTPUT=$(python src/train_classifier.py --epochs ${CLASSIFIER_EPOCHS} --ood ${OOD_DATASET})
CLASSIFIER_DIR=$(echo "${CLASSIFIER_OUTPUT}" | grep "CLASSIFIER_RESULTS_DIR=" | cut -d '=' -f2)
echo "Classifier training finished. Results dir: ${CLASSIFIER_DIR}"

# 2. Train auto-encoder
echo "Starting autoencoder training..."
AUTOENCODER_OUTPUT=$(python src/train_autoencoder.py --epochs ${AUTOENCODER_EPOCHS} -b ${AE_BOTTLENECK} --ood ${OOD_DATASET})
AUTOENCODER_DIR=$(echo "${AUTOENCODER_OUTPUT}" | grep "AUTOENCODER_RESULTS_DIR=" | cut -d '=' -f2)
echo "Autoencoder training finished. Results dir: ${AUTOENCODER_DIR}"

# Create config directory if it doesn't exist
mkdir -p configs

# Save parameters to a file for run_eval.sh to use
CONFIG_FILE="configs/last_run_params.env"
echo "Saving run parameters to ${CONFIG_FILE}"
echo "OOD_DATASET=${OOD_DATASET}" > ${CONFIG_FILE}
echo "AE_BOTTLENECK=${AE_BOTTLENECK}" >> ${CONFIG_FILE}
echo "CLASSIFIER_DIR=${CLASSIFIER_DIR}" >> ${CONFIG_FILE}
echo "AUTOENCODER_DIR=${AUTOENCODER_DIR}" >> ${CONFIG_FILE}

echo "Training complete. Parameters saved."