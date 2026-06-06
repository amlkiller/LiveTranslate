# LiveTranslate - One-click installer
# Usage: Double-click install.bat (or run: powershell -ExecutionPolicy Bypass -File install.ps1)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[$((Get-Date).ToString('HH:mm:ss'))] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  WARN: $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  ERROR: $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "   LiveTranslate Installer" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

# ── Step 1: Find Python ──
Write-Step "Detecting Python..."

function Find-Python {
    # 3.13+ rejected: no ctranslate2 cp313 wheels (#15), strict SSL breaks torch.hub (#20)
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>&1
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) {
                $exe = $exe.Trim()
                $ver = & $exe --version 2>&1
                Write-Ok "Found $ver ($exe)"
                return $exe
            }
        } catch {}
    }
    # Fall back to plain commands, rejecting unsupported versions.
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 10 -and $minor -le 12) {
                    Write-Ok "Found $ver ($cmd)"
                    return $cmd
                } elseif ($major -eq 3 -and $minor -ge 13) {
                    Write-Warn "$ver is too new (faster-whisper/SSL require 3.10-3.12)"
                } else {
                    Write-Warn "$ver is too old (need 3.10-3.12)"
                }
            }
        } catch {}
    }
    return $null
}

$PythonCmd = Find-Python

if (-not $PythonCmd) {
    Write-Warn "No supported Python (3.10-3.12) found"

    # Try to install via winget
    $hasWinget = $false
    try {
        $null = & winget --version 2>&1
        if ($LASTEXITCODE -eq 0) { $hasWinget = $true }
    } catch {}

    if ($hasWinget) {
        Write-Host ""
        Write-Host "  Python can be installed automatically via winget." -ForegroundColor White
        $answer = Read-Host "  Install Python 3.12 now? [Y/n]"
        if ($answer -eq "" -or $answer -match "^[Yy]") {
            Write-Step "Installing Python 3.12 via winget..."
            & winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                Write-Err "winget install failed"
                Read-Host "Press Enter to exit"
                exit 1
            }

            # Refresh PATH to pick up newly installed Python
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

            $PythonCmd = Find-Python
            if (-not $PythonCmd) {
                Write-Err "Python installed but not found in PATH. Please close this window, reopen, and run install.bat again."
                Read-Host "Press Enter to exit"
                exit 1
            }
        } else {
            Write-Err "Python 3.10-3.12 is required. Please install from https://www.python.org/downloads/"
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Err "Python 3.10-3.12 not found and winget is not available."
        Write-Host "  Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ── Step 2: Create venv ──
Write-Step "Creating virtual environment..."

# Validate existing venv: even if python.exe is present, the venv could be
# half-built, created with a different Python, or corrupted (see issue #18).
# If it's broken, recreate it; otherwise reuse it.
function Test-VenvHealthy {
    param([string]$VenvPythonExe)
    if (-not (Test-Path $VenvPythonExe)) { return $false }
    try {
        $ver = & $VenvPythonExe --version 2>&1
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($ver -notmatch "Python \d+\.\d+") { return $false }
        return $true
    } catch {
        return $false
    }
}

$VenvPython = ".venv\Scripts\python.exe"
if (Test-Path ".venv") {
    if (Test-VenvHealthy $VenvPython) {
        Write-Ok "Existing venv is healthy, reusing"
    } else {
        Write-Warn "Existing venv is broken or incomplete, recreating..."
        Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
        & $PythonCmd -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to create venv"
            Read-Host "Press Enter to exit"
            exit 1
        }
        Write-Ok "Created .venv"
    }
} else {
    & $PythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create venv"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Ok "Created .venv"
}

$Pip = ".venv\Scripts\pip.exe"
$Python = ".venv\Scripts\python.exe"

# Upgrade pip first
Write-Step "Upgrading pip..."
& $Python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip upgrade failed (non-critical, continuing with current pip)"
} else {
    Write-Ok "pip upgraded"
}

# ── Step 3: Detect GPU ──
Write-Step "Detecting GPU..."

$HasNvidia = $false
$CudaVer = "cu126"
try {
    $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        $HasNvidia = $true
        Write-Ok "NVIDIA GPU detected: $($gpu.Trim())"

        # Detect compute capability to choose CUDA version
        # Blackwell (sm_120, compute_cap >= 12.0) requires cu128
        $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $cc) {
            $ccVal = [double]($cc.Trim())
            if ($ccVal -ge 12.0) {
                $CudaVer = "cu128"
                Write-Ok "Blackwell+ architecture (sm_$($cc.Trim() -replace '\.','')) detected, using CUDA 12.8"
            } else {
                Write-Ok "Compute capability $($cc.Trim()), using CUDA 12.6"
            }
        }
    }
} catch {}

if (-not $HasNvidia) {
    Write-Warn "No NVIDIA GPU detected, will install CPU-only PyTorch"
}

# Let user choose
Write-Host ""
if ($HasNvidia) {
    $cudaLabel = if ($CudaVer -eq "cu128") { "CUDA 12.8" } else { "CUDA 12.6" }
    Write-Host "  [1] $cudaLabel (recommended for your NVIDIA GPU)" -ForegroundColor White
    Write-Host "  [2] CPU only" -ForegroundColor White
    $choice = Read-Host "  Select PyTorch version [1]"
    if ($choice -eq "2") { $HasNvidia = $false }
} else {
    Write-Host "  [1] CPU only" -ForegroundColor White
    Write-Host "  [2] CUDA (if you have NVIDIA GPU)" -ForegroundColor White
    $choice = Read-Host "  Select PyTorch version [1]"
    if ($choice -eq "2") { $HasNvidia = $true }
}

# ── Step 4: Install PyTorch ──
Write-Step "Installing PyTorch (this may take a few minutes)..."

if ($HasNvidia) {
    Write-Host "  Using index: $CudaVer" -ForegroundColor Gray
    & $Pip install torch torchaudio --index-url https://download.pytorch.org/whl/$CudaVer
} else {
    & $Pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Err "PyTorch installation failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "PyTorch installed"

# ── Step 5: Install dependencies ──
Write-Step "Installing dependencies from requirements.txt..."

& $Pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to install dependencies"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "Dependencies installed"

# ── Step 6: Install FunASR (no-deps) ──
Write-Step "Installing FunASR (--no-deps)..."

& $Pip install funasr --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-Warn "FunASR installation failed (non-critical, SenseVoice engine may not work)"
} else {
    Write-Ok "FunASR installed"
}

# ── Step 7: Install pysbd for incremental ASR ──
Write-Step "Installing pysbd..."

& $Pip install pysbd
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pysbd installation failed (incremental ASR may not work)"
} else {
    Write-Ok "pysbd installed"
}

# ── Done ──
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start LiveTranslate:" -ForegroundColor White
Write-Host "    Double-click start.bat" -ForegroundColor Yellow
Write-Host "    or run: .venv\Scripts\python.exe main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  First launch will download ASR models (~1GB)." -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
