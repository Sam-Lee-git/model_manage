# install.ps1 — install model-manager and make `mm` available in PATH permanently
# Usage: .\install.ps1
# Run from the repo root directory.

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==> Installing model-manager from $RepoDir"
pip install -e $RepoDir

# Find where pip installed the mm.exe script
$ScriptsDir = python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
$MmExe = Join-Path $ScriptsDir "mm.exe"

if (-not (Test-Path $MmExe)) {
    Write-Error "mm.exe not found at $MmExe — install may have failed."
    exit 1
}

# Check if already in PATH
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -like "*$ScriptsDir*") {
    Write-Host "==> $ScriptsDir is already in PATH"
} else {
    # Add to user PATH permanently (no admin required)
    [Environment]::SetEnvironmentVariable("PATH", "$ScriptsDir;$UserPath", "User")
    Write-Host "==> Added to PATH (User): $ScriptsDir"

    # Also apply to the current PowerShell session immediately
    $env:PATH = "$ScriptsDir;$env:PATH"
    Write-Host "==> PATH updated for this session too"
}

Write-Host ""
Write-Host "==> Done. Verifying..."
& $MmExe --version

Write-Host ""
Write-Host "You can now run: mm"
Write-Host "(New terminals will also have mm available — no need to rerun this script)"
