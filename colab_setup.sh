#!/bin/bash
# SatMAE Colab Setup Script
# Quick setup for Google Colab environment

echo "🚀 SatMAE Colab Setup"
echo "===================="

# Install required packages
echo "📦 Installing packages..."
pip install timm==0.3.2 --quiet
pip install rasterio --quiet  
pip install wandb --quiet

# Clone repository (if not already cloned)
if [ ! -d "SatMAE" ]; then
    echo "📥 Cloning SatMAE repository..."
    git clone https://github.com/pvinnbru/SatMAE.git
    cd SatMAE
else
    echo "✅ SatMAE repository already exists"
    cd SatMAE
fi

# Create necessary directories
mkdir -p data checkpoints data_splits results

# Download EuroSAT dataset
echo "📊 Downloading EuroSAT dataset..."
cd data
if [ ! -f "EuroSATallBands.zip" ]; then
    wget -q --show-progress https://madm.dfki.de/files/sentinel/EuroSATallBands.zip
    echo "📂 Extracting dataset..."
    unzip -q EuroSATallBands.zip
    echo "✅ Dataset ready!"
else
    echo "✅ Dataset already downloaded"
fi

cd ..

echo ""
echo "🎯 Setup complete! Next steps:"
echo "1. Upload your pretrained checkpoint to checkpoints/"
echo "2. Run the data preprocessing cells"
echo "3. Start training!"
echo ""
echo "💡 Tip: Use GPU runtime for faster training"
