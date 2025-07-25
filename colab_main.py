"""
SatMAE Google Colab Training Script
Simplified setup and training for Google Colab environment
"""

import os
import sys
import subprocess
import zipfile
import urllib.request
from pathlib import Path

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️ No GPU detected!")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False

def install_packages():
    """Install required packages"""
    packages = [
        "timm==0.3.2",
        "rasterio", 
        "wandb",
        "tensorboard",
        "gdown"  # For Google Drive downloads
    ]
    
    print("Installing required packages...")
    for package in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                      check=True)
    print("✅ All packages installed!")

def download_dataset(google_drive_id=None):
    """Download and extract EuroSAT dataset"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    eurosat_file = data_dir / "EuroSATallBands.zip"
    
    if not eurosat_file.exists():
        if google_drive_id:
            print("Downloading EuroSAT dataset from Google Drive...")
            try:
                import gdown
                gdown.download(f"https://drive.google.com/uc?id={google_drive_id}", 
                             str(eurosat_file), quiet=False)
                print("✅ Google Drive download complete!")
            except Exception as e:
                print(f"❌ Google Drive download failed: {e}")
                print("Falling back to original server...")
                google_drive_id = None
        
        if not google_drive_id:
            print("Downloading EuroSAT dataset from original server...")
            print("This may take a few minutes (~2.8GB)...")
            eurosat_url = "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
            urllib.request.urlretrieve(eurosat_url, eurosat_file)
            print("✅ Download complete!")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(eurosat_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✅ Dataset extracted!")
    else:
        print("✅ Dataset already downloaded")

def create_data_splits():
    """Create train/val splits and subsets"""
    import random
    from glob import glob
    
    def generate_split_txt(root_folder, out_txt_path, split_ratio=0.8, seed=42):
        class_names = sorted(os.listdir(root_folder))
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        all_samples = []
        for cls in class_names:
            tif_paths = glob(os.path.join(root_folder, cls, "*.tif"))
            for path in tif_paths:
                all_samples.append(f"{path} {class_to_idx[cls]}")

        if not all_samples:
            print(f"⚠️ No files found in: {root_folder}")
            return

        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        with open(out_txt_path.replace(".txt", "_train.txt"), "w") as f:
            f.write("\\n".join(train_samples))
        with open(out_txt_path.replace(".txt", "_val.txt"), "w") as f:
            f.write("\\n".join(val_samples))

        print(f"✅ Created splits: {len(train_samples)} train, {len(val_samples)} val")

    def subsample_txt_file(input_path, output_prefix, percentages=[10, 25, 50, 75], seed=42):
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        random.seed(seed)
        random.shuffle(lines)
        
        for p in percentages:
            count = int(len(lines) * (p / 100))
            subset = lines[:count]
            out_path = f"{output_prefix}_{p}.txt"
            with open(out_path, 'w') as f_out:
                f_out.writelines(subset)
            print(f"Created {p}% subset: {count} samples")

    # Create directories
    Path("data_splits").mkdir(exist_ok=True)
    
    # Generate splits
    print("Creating train/val splits...")
    generate_split_txt("data/EuroSATallBands", "data_splits/eurosat_ms.txt")
    
    print("Creating training subsets...")
    subsample_txt_file("data_splits/eurosat_ms_train.txt", "data_splits/eurosat_ms_train")

def run_training(data_percentage=10):
    """Run SatMAE training"""
    import torch
    
    # Determine batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 15:
            batch_size, accum_iter = 16, 8
        elif gpu_memory_gb >= 11:
            batch_size, accum_iter = 8, 16
        else:
            batch_size, accum_iter = 4, 32
    else:
        batch_size, accum_iter = 4, 32
    
    print(f"Using batch_size={batch_size}, accum_iter={accum_iter}")
    
    # Create results directory
    output_dir = f"results/eurosat_ms_{data_percentage}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training command
    cmd = [
        "python", "main_finetune.py",
        "--model_type", "group_c",
        "--model", "vit_large_patch16", 
        "--dataset_type", "euro_sat",
        "--train_path", f"data_splits/eurosat_ms_train_{data_percentage}.txt",
        "--test_path", "data_splits/eurosat_ms_val.txt",
        "--finetune", "checkpoints/pretrain-vit-large-e199.pth",
        "--input_size", "96", "--patch_size", "8",
        "--batch_size", str(batch_size), "--accum_iter", str(accum_iter),
        "--epochs", "30", "--blr", "2e-4",
        "--weight_decay", "0.05",
        "--drop_path", "0.2", "--reprob", "0.25", "--mixup", "0.8", "--cutmix", "1.0",
        "--dropped_bands", "0", "9", "10",
        "--num_workers", "2",
        "--output_dir", output_dir,
        "--log_dir", output_dir
    ]
    
    print(f"Starting training with {data_percentage}% of data...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")

def main():
    """Main setup and training function"""
    print("🚀 SatMAE Colab Setup")
    print("=" * 50)
    
    # Check environment
    has_gpu = check_gpu()
    if not has_gpu:
        print("⚠️ Consider enabling GPU runtime for faster training")
    
    # Install packages
    install_packages()
    
    # Download data
    print("\n" + "="*50)
    print("📊 DATASET SETUP")
    print("="*50)
    
    # You can provide your Google Drive file ID here for faster downloads
    google_drive_id = None  # Replace with your Google Drive file ID
    # google_drive_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Uncomment and set your ID
    
    if google_drive_id:
        print("🚀 Using Google Drive for faster download!")
    else:
        print("📡 Using original server (slower but works)")
    
    download_dataset(google_drive_id)
    
    # Create data splits
    create_data_splits()
    
    # Check for checkpoint
    checkpoint_path = "checkpoints/pretrain-vit-large-e199.pth"
    if not os.path.exists(checkpoint_path):
        print("\\n⚠️ Please upload your pretrained checkpoint:")
        print(f"   Expected location: {checkpoint_path}")
        print("   Use Colab's file upload feature")
        return
    
    print("\\n🚀 Setup complete! Ready for training.")
    print("\\nTo start training, run:")
    print("run_training(data_percentage=10)")

if __name__ == "__main__":
    main()
