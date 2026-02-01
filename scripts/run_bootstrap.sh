#!/bin/bash
#
# Run bootstrap training for VAE models
#
# Usage:
#   ./run_bootstrap.sh [N_BOOTSTRAP] [EPOCHS]
#
# Example:
#   ./run_bootstrap.sh 100 100
#

set -e

N_BOOTSTRAP=${1:-100}
EPOCHS=${2:-100}
DATA="data/vae_training_data_v2_20cm.csv"
OUTPUT_DIR="models/bootstrap"

echo "=============================================="
echo "BOOTSTRAP TRAINING"
echo "=============================================="
echo "N bootstrap: $N_BOOTSTRAP"
echo "Epochs: $EPOCHS"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"
echo "=============================================="
echo

# Create output directories
mkdir -p "$OUTPUT_DIR/unsupervised"
mkdir -p "$OUTPUT_DIR/semisupervised"

# Run bootstrap iterations
for i in $(seq 1 $N_BOOTSTRAP); do
    echo "========================================"
    echo "Bootstrap iteration $i / $N_BOOTSTRAP"
    echo "========================================"
    
    SEED=$((42 + i))
    
    # Train unsupervised
    echo "Training unsupervised VAE (seed=$SEED)..."
    python scripts/train_vae.py \
        --model unsupervised \
        --data "$DATA" \
        --output "$OUTPUT_DIR/unsupervised/model_$i.pt" \
        --epochs $EPOCHS \
        --seed $SEED \
        --beta 1.0 \
        2>&1 | tail -5
    
    # Train semi-supervised
    echo "Training semi-supervised VAE (seed=$SEED)..."
    python scripts/train_vae.py \
        --model semisupervised \
        --data "$DATA" \
        --output "$OUTPUT_DIR/semisupervised/model_$i.pt" \
        --epochs $EPOCHS \
        --seed $SEED \
        --beta 1.0 \
        --alpha 0.1 \
        2>&1 | tail -5
    
    echo
done

echo "=============================================="
echo "BOOTSTRAP COMPLETE"
echo "=============================================="
echo "Unsupervised models: $OUTPUT_DIR/unsupervised/"
echo "Semi-supervised models: $OUTPUT_DIR/semisupervised/"
echo "=============================================="
