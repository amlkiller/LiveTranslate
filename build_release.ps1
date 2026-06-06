# LiveTranslate - Portable release builder
# Produces a self-contained zip that runs without a system Python install.
# First launch uses a bundled uv to fetch Python 3.12 + GPU-aware dependencies.

param([string]$Version = "")

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[BUILD] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }

$UvUrl  = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$OutDir = Join-Path $ProjectDir "release"
$Stage  = Join-Path $OutDir "LiveTranslate"

$Sha   = (& git rev-parse --short HEAD).Trim()
$Stamp = Get-Date -Format "yyyyMMdd"
$Tag   = if ($Version) { $Version } else { "$Stamp-$Sha" }
$ZipPath = Join-Path $OutDir "LiveTranslate-portable-$Tag.zip"

# Files only needed for the git-clone workflow; the portable zip ships its own launcher.
$DropList = @("install.bat", "install.ps1", "update.bat", "start.bat",
              "build_release.ps1", "CLAUDE.md", "test_audio.py", ".gitignore", "screenshot")

# ── 1. Clean staging ──
Write-Step "Preparing staging directory..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

# ── 2. Export tracked source via git archive ──
Write-Step "Exporting source (git archive HEAD)..."
$Tar = Join-Path $OutDir "src.tar"
& git archive --format=tar -o $Tar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
& tar -x -f $Tar -C $Stage
Remove-Item $Tar
foreach ($d in $DropList) {
    $p = Join-Path $Stage $d
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Write-Ok "Source exported"

# ── 3. Download and bundle uv ──
Write-Step "Downloading uv..."
$UvZip = Join-Path $OutDir "uv.zip"
Invoke-WebRequest -Uri $UvUrl -OutFile $UvZip
$ToolsDir = Join-Path $Stage "tools"
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
Expand-Archive -Path $UvZip -DestinationPath $ToolsDir -Force
Remove-Item $UvZip
$UvExe = Join-Path $ToolsDir "uv.exe"
if (-not (Test-Path $UvExe)) { throw "uv.exe not found after extract" }
Write-Ok ("Bundled " + (& $UvExe --version))

# ── 4. Write portable launcher + bootstrap ──
Write-Step "Writing launcher..."
$StartBat = @'
@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up environment. This downloads Python and dependencies and may take several minutes...
    powershell -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
    if errorlevel 1 (
        echo.
        echo [ERROR] Setup failed. See messages above.
        pause
        exit /b 1
    )
)
echo Starting LiveTranslate...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
'@
Set-Content -Path (Join-Path $Stage "start.bat") -Value $StartBat -Encoding ASCII

$Bootstrap = @'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Uv = Join-Path $Root "tools\uv.exe"
$env:UV_LINK_MODE = "copy"

Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
& $Uv venv --python 3.12 --managed-python .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create venv" -ForegroundColor Red; exit 1 }
$Py = ".venv\Scripts\python.exe"

# Blackwell (sm_120+) needs cu128; older NVIDIA uses cu126; no GPU falls back to CPU
$Index = "https://download.pytorch.org/whl/cpu"
try {
    $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $cc) {
        $cap = [double]($cc.Trim() -split "`n")[0]
        if ($cap -ge 12.0) { $Index = "https://download.pytorch.org/whl/cu128" }
        else { $Index = "https://download.pytorch.org/whl/cu126" }
        Write-Host "NVIDIA GPU detected (compute $cap), using $Index" -ForegroundColor Green
    }
} catch {}
if ($Index -like "*cpu*") { Write-Host "No NVIDIA GPU detected, installing CPU-only PyTorch" -ForegroundColor Yellow }

Write-Host "Installing PyTorch (this may take a while)..." -ForegroundColor Cyan
& $Uv pip install --python $Py torch torchaudio --index-url $Index
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch install failed" -ForegroundColor Red; exit 1 }

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $Uv pip install --python $Py -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Dependency install failed" -ForegroundColor Red; exit 1 }

& $Uv pip install --python $Py funasr --no-deps
& $Uv pip install --python $Py pysbd

Write-Host "Setup complete." -ForegroundColor Green
'@
Set-Content -Path (Join-Path $Stage "bootstrap.ps1") -Value $Bootstrap -Encoding ASCII
Write-Ok "Launcher written"

# ── 5. Zip ──
Write-Step "Creating archive..."
Compress-Archive -Path $Stage -DestinationPath $ZipPath -Force
$sizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Ok "Created $ZipPath ($sizeMb MB)"
Write-Host "`nDone. Distribute: $ZipPath" -ForegroundColor Green
