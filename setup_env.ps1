# setup_env.ps1
# Automates the setup of a local Miniconda and Python environment inside the project directory.

$InstallDir = "$PSScriptRoot\miniconda3"
$InstallerPath = "$PSScriptRoot\miniconda_installer.exe"
$PythonPath = "$InstallDir\python.exe"

# 1. Download Miniconda if not already downloaded and python doesn't exist
if (-not (Test-Path $PythonPath)) {
    if (-not (Test-Path $InstallerPath)) {
        Write-Host "Downloading Miniconda (this may take a moment)..." -ForegroundColor Cyan
        $Url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        Invoke-WebRequest -Uri $Url -OutFile $InstallerPath -UserAgent "Mozilla/5.0"
        Write-Host "Download complete." -ForegroundColor Green
    }

    # 2. Run silent installer
    Write-Host "Installing Miniconda silently to $InstallDir..." -ForegroundColor Cyan
    $Arguments = "/S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=$InstallDir"
    $Process = Start-Process -FilePath $InstallerPath -ArgumentList $Arguments -Wait -NoNewWindow -PassThru

    if ($Process.ExitCode -eq 0) {
        Write-Host "Miniconda installed successfully." -ForegroundColor Green
    } else {
        Write-Error "Miniconda installation failed with exit code $($Process.ExitCode)."
        exit $Process.ExitCode
    }

    # Clean up installer
    if (Test-Path $InstallerPath) {
        Remove-Item $InstallerPath -Force
    }
} else {
    Write-Host "Miniconda is already installed at $InstallDir." -ForegroundColor Green
}

# 3. Upgrade pip and install required packages
Write-Host "Installing python packages from requirements.txt..." -ForegroundColor Cyan
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install -r "$PSScriptRoot\requirements.txt"

Write-Host "Installing PyTorch..." -ForegroundColor Cyan
# Install CPU version of PyTorch for maximum compatibility (unless GPU is specifically required, CPU is safer for a general setup, but standard pip install torch will download the default which has CUDA if available)
& $PythonPath -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "You can run python scripts using: .\miniconda3\python.exe <script.py>" -ForegroundColor Yellow
