# SatMAE on Google Colab

This guide helps you run SatMAE finetuning on Google Colab with free GPU access.

## 🚀 Quick Start

### Option 1: Use the Jupyter Notebook (Recommended)
1. Upload `SatMAE_Colab.ipynb` to Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Upload your pretrained checkpoint when prompted

### Option 2: Use the Python Script
1. In a new Colab notebook, run:
```python
# Clone the repository
!git clone https://github.com/pvinnbru/SatMAE.git
%cd SatMAE

# Run the setup script
exec(open('colab_main.py').read())
```

2. Upload your checkpoint to `checkpoints/pretrain-vit-large-e199.pth`

3. Start training:
```python
run_training(data_percentage=10)
```

## 📋 What You Need

1. **Pretrained Checkpoint**: Upload your `pretrain-vit-large-e199.pth` file
2. **EuroSAT Dataset**: Either upload to Google Drive (recommended) or let it download automatically
3. **GPU Runtime**: Enable GPU in Colab settings
4. **Time**: ~30-60 minutes for 10% data training

### 🚀 Google Drive Setup (Recommended)

For faster downloads, upload your EuroSAT dataset to Google Drive:

1. **Upload** `EuroSATallBands.zip` to Google Drive
2. **Share**: Right-click → Share → "Anyone with the link"
3. **Copy** the sharing URL: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
4. **Extract** the `FILE_ID` from the URL
5. **Update** the notebook: Replace `YOUR_GOOGLE_DRIVE_FILE_ID` with your actual `FILE_ID`

**Benefits:**
- ⚡ 3-5x faster downloads
- 🔄 More reliable connection
- 📊 Better progress tracking
- 🔗 Reusable across sessions

## 🔧 Features

- ✅ Automatic package installation (timm, rasterio, wandb, tensorboard, gdown)
- ✅ Google Drive integration for fast dataset downloads
- ✅ Automatic EuroSAT dataset download and extraction
- ✅ Automatic data preprocessing and splits
- ✅ GPU memory optimization (adapts batch size to available memory)
- ✅ TensorBoard integration for monitoring
- ✅ Results packaging for download
- ✅ Multiple data percentage experiments (10%, 25%, 50%, 75%)

## 📊 Expected Results

Training on 10% of EuroSAT data should complete in:
- **T4 GPU**: ~45-60 minutes
- **V100 GPU**: ~20-30 minutes  
- **A100 GPU**: ~15-20 minutes

## 💾 GPU Memory Requirements

The script automatically adjusts batch sizes:
- **15GB+ (A100, V100)**: batch_size=16, accum_iter=8
- **11-15GB (T4)**: batch_size=8, accum_iter=16
- **<11GB**: batch_size=4, accum_iter=32

## 📁 File Structure

```
SatMAE/
├── SatMAE_Colab.ipynb          # Main Colab notebook
├── colab_main.py               # Python setup script
├── colab_setup.sh              # Bash setup script
├── main_finetune.py            # Training script
├── data/                       # EuroSAT dataset
├── data_splits/                # Train/val splits
├── checkpoints/                # Upload your checkpoint here
└── results/                    # Training outputs
```

## 🎯 Training Command

The script runs this command automatically:
```bash
python main_finetune.py \
  --model_type group_c \
  --model vit_large_patch16 \
  --dataset_type euro_sat \
  --train_path data_splits/eurosat_ms_train_10.txt \
  --test_path data_splits/eurosat_ms_val.txt \
  --finetune checkpoints/pretrain-vit-large-e199.pth \
  --input_size 96 --patch_size 8 \
  --batch_size 8 --accum_iter 16 \
  --epochs 30 --blr 2e-4 \
  --weight_decay 0.05 \
  --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --dropped_bands 0 9 10 \
  --num_workers 2 \
  --output_dir results/eurosat_ms_10 \
  --log_dir results/eurosat_ms_10
```

## 📈 Monitoring

- **TensorBoard**: Monitor training progress in real-time
- **Logs**: Training logs saved to results directory
- **Checkpoints**: Model checkpoints saved for each epoch

## 💡 Tips

1. **Enable GPU**: Essential for reasonable training times
2. **Save Results**: Download the results archive before session ends
3. **Experiment**: Try different data percentages (10%, 25%, 50%, 75%)
4. **Monitor**: Use TensorBoard to watch training progress
5. **Backup**: Colab sessions timeout after 12 hours

## 🆘 Troubleshooting

**"No GPU detected"**: Enable GPU runtime in Colab settings
**"Checkpoint not found"**: Upload your .pth file to checkpoints/ folder
**"Out of memory"**: Script should auto-adjust, but you can manually reduce batch_size
**"Dataset download fails"**: Check internet connection, dataset is ~2.8GB

## 🎉 Success!

After training completes, you'll have:
- Trained model checkpoints
- Training logs and metrics
- TensorBoard visualizations
- Downloadable results archive

Perfect for validating your SatMAE pipeline without server limitations!
