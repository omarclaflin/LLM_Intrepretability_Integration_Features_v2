param (
    [string]$ModelId = "openlm-research/open_llama_3b",
    [string]$OutputDir = "..\models\open_llama_3b"
)

$ErrorActionPreference = "Stop"
Write-Host "Starting download of model: $ModelId" -ForegroundColor Cyan

# Check if huggingface_hub is installed
$hubInstalled = python -c "import importlib.util; print(importlib.util.find_spec('huggingface_hub') is not None)" 2>$null
if ($hubInstalled -ne "True") {
    Write-Host "Installing huggingface_hub..." -ForegroundColor Yellow
    pip install huggingface_hub
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install huggingface_hub"
        exit 1
    }
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    Write-Host "Creating directory: $OutputDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Download the model using Python directly instead of the CLI
Write-Host "Downloading model files to: $OutputDir" -ForegroundColor Cyan
Write-Host "This may take some time depending on your internet speed..." -ForegroundColor Yellow

$pythonScript = @"
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='$ModelId',
    local_dir='$OutputDir', 
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {path}')
"@

python -c "$pythonScript"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download model"
    exit 1
}

# Verify download
$configPath = Join-Path $OutputDir "config.json"
if (Test-Path $configPath) {
    Write-Host "Model downloaded successfully to: $OutputDir" -ForegroundColor Green
    
    # Get size of downloaded files
    $size = (Get-ChildItem $OutputDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "Total model size: $([math]::Round($size, 2)) GB" -ForegroundColor Cyan
    
    # List main model files
    Write-Host "Main model files:" -ForegroundColor Cyan
    Get-ChildItem $OutputDir -File | 
        Where-Object { $_.Extension -eq ".bin" -or $_.Extension -eq ".json" -or $_.Extension -eq ".model" } | 
        Select-Object Name, @{Name="Size (MB)"; Expression={[math]::Round($_.Length / 1MB, 2)}} | 
        Format-Table -AutoSize
} else {
    Write-Warning "Download appears incomplete. config.json not found in $OutputDir"
}

Write-Host "Loading example:"
Write-Host "from transformers import LlamaTokenizer, LlamaForCausalLM" -ForegroundColor DarkGray
Write-Host "tokenizer = LlamaTokenizer.from_pretrained('$OutputDir', use_fast=False)" -ForegroundColor DarkGray
Write-Host "model = LlamaForCausalLM.from_pretrained('$OutputDir', torch_dtype='auto', device_map='auto')" -ForegroundColor DarkGray