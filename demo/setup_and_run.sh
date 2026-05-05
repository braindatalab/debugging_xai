#!/bin/bash

# Exit on first error
set -e

FORCE=0
if [ "$1" == "--force" ]; then
    FORCE=1
fi

echo "=== 1. Setting up Demo Data ==="
mkdir -p demo/images/cat
mkdir -p demo/images/dog

# Check if the demo images already exist
cat_count=$(ls -1q demo/images/cat/ 2>/dev/null | wc -l)
dog_count=$(ls -1q demo/images/dog/ 2>/dev/null | wc -l)

if [ "$cat_count" -ge 20 ] && [ "$dog_count" -ge 20 ]; then
    echo "Found demo images in demo/images/, proceeding with existing data."
elif [ -d "images/cat" ] && [ -d "images/dog" ]; then
    echo "Copying 20 sample images from the full dataset..."
    find images/cat -maxdepth 1 -type f | head -n 20 | xargs -I {} cp {} demo/images/cat/
    find images/dog -maxdepth 1 -type f | head -n 20 | xargs -I {} cp {} demo/images/dog/
else
    echo "Error: Could not find 20 images in demo/images/, and the original images/ directory was not found."
    echo "Please place at least 20 sample images in demo/images/cat/ and demo/images/dog/"
    exit 1
fi

echo "=== 2. Generating Watermarked Data ==="
if [ "$FORCE" -eq 1 ] || [ ! -f "demo/artifacts/split_0_suppressor_train.pkl" ]; then
    # We use N=20 to only process the 20 images per class we just copied
    python -m watermarks.generate_watermarks \
        --split-index 0 \
        --cats-dir demo/images/cat \
        --dogs-dir demo/images/dog \
        --outdir demo/artifacts \
        --watermark "watermark banner.jpg" \
        --N 20
else
    echo "Generated data already exists in demo/artifacts/. Skipping... (use --force to override)"
fi

echo "=== 3. Training Models ==="
if [ "$FORCE" -eq 1 ] || [ ! -f "demo/models/cnn_suppressor_split0_seed12031212.pt" ]; then
    # Train for 2 epochs on the demo data with a small batch size
    python -m watermarks.train_watermarks_server \
        --split-index 0 \
        --base all \
        --artifacts-dir demo/artifacts \
        --model-dir demo/models \
        --batch-size 8 \
        --epochs 2
else
    echo "Trained models already exist in demo/models/. Skipping... (use --force to override)"
fi

echo "=== 4. Evaluating Metrics ==="
if [ "$FORCE" -eq 1 ] || [ ! -f "demo/results/energies/energy_water_conf_pred_split0_seed12031212.pickle" ]; then
    python -m watermarks.calculate_energy \
        --split-index 0 \
        --seed-index 0 \
        --artifacts-dir demo/artifacts \
        --models-dir demo/models \
        --energies-dir demo/results/energies \
        --explanations-dir demo/results/explanations \
        --limit 10
else
    echo "Evaluation results already exist in demo/results/. Skipping... (use --force to override)"
fi

echo "=== Demo completed successfully! ==="
echo "Results are saved in demo/results/"
