# setup-llm-interpretability.ps1
$ErrorActionPreference = "Stop"

Write-Host "Setting up LLM Interpretability Environment" -ForegroundColor Green

# Check Python version
$pythonVersion = (python --version) 2>&1
if (-not ($pythonVersion -like "*Python 3.13*")) {
    Write-Warning "Python 3.13 is recommended but not found. Current version: $pythonVersion"
    $proceed = Read-Host "Continue anyway? (y/n)"
    if ($proceed -ne "y") {
        exit 1
    }
}

# Create and activate virtual environment
if (-not (Test-Path "llm-venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv llm-venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\llm-venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers and related libraries
Write-Host "Installing Transformers and related libraries..." -ForegroundColor Cyan
pip install transformers datasets accelerate

# Install quantization libraries
Write-Host "Installing quantization libraries..." -ForegroundColor Cyan
pip install bitsandbytes optimum auto-gptq

# Install interpretability tools
Write-Host "Installing interpretability tools..." -ForegroundColor Cyan
pip install captum ecco
pip install git+https://github.com/neelnanda-io/TransformerLens.git

# Install visualization libraries
Write-Host "Installing visualization libraries..." -ForegroundColor Cyan
pip install matplotlib seaborn plotly

# Install utility libraries
Write-Host "Installing utility libraries..." -ForegroundColor Cyan
pip install tqdm numpy pandas scikit-learn

# Test CUDA availability using a temporary file instead of inline command
Write-Host "Testing CUDA availability..." -ForegroundColor Cyan

$tempFile = "cuda_test_temp.py"

@"
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices available")
"@ | Out-File -FilePath $tempFile -Encoding utf8

python $tempFile
Remove-Item $tempFile

# Save requirements file for future reference
pip freeze > requirements.txt
Write-Host "Requirements saved to requirements.txt" -ForegroundColor Green

Write-Host "Setup complete! Activate the environment in future sessions with: .\llm-venv\Scripts\Activate.ps1" -ForegroundColor Green