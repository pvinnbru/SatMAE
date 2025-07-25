#!/bin/bash
# SatMAE Environment Setup Script
# For servers with 6GB space constraint

echo "SatMAE Environment Setup"
echo "========================"

# Check available space
echo "Checking available disk space..."
df -h .

# Check if GPU is available
echo -e "\nChecking for CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    USE_GPU=true
else
    echo "No GPU detected, using CPU version"
    USE_GPU=false
fi

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv satmae_env
source satmae_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages based on GPU availability
if [ "$USE_GPU" = true ]; then
    echo -e "\nInstalling GPU version (CUDA 11.3)..."
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements_gpu.txt
else
    echo -e "\nInstalling CPU version..."
    pip install -r requirements.txt
fi

echo -e "\nSetup complete!"
echo "To activate the environment: source satmae_env/bin/activate"
echo "To run finetuning, update your command to use absolute paths:"
echo ""
echo "python main_finetune.py \\"
echo "  --model_type group_c \\"
echo "  --model vit_large_patch16 \\"
echo "  --dataset_type euro_sat \\"
echo "  --train_path data_splits/eurosat_ms_train_10.txt \\"
echo "  --test_path data_splits/eurosat_ms_val.txt \\"
echo "  --finetune checkpoints/pretrain-vit-large-e199.pth \\"
echo "  --input_size 96 --patch_size 8 \\"
echo "  --batch_size 8 --accum_iter 16 \\"
echo "  --epochs 30 --blr 2e-4 \\"
echo "  --weight_decay 0.05 \\"
echo "  --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \\"
echo "  --dropped_bands 0 9 10 \\"
echo "  --num_workers 4 \\"
echo "  --output_dir results/eurosat_ms_10 \\"
echo "  --log_dir results/eurosat_ms_10"
