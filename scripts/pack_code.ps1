# Pack the committed code into lowlight-code.zip for Google Colab (no GitHub needed).
#
# Usage (PowerShell, from the repo root):   .\scripts\pack_code.ps1
# Then drag lowlight-code.zip into  My Drive/lowlight-seg/code/  (replace the old one).
#
# Only COMMITTED files are packed, and the commit id is stored inside the zip, so every Colab
# result can be traced back to the exact code version. Data, weights and .venv are never included.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$changes = git status --porcelain
if ($changes) {
    Write-Warning "You have uncommitted changes. They will NOT be in the zip. Commit first if you need them."
}

git archive --format=zip -o lowlight-code.zip HEAD
$commit = git rev-parse --short HEAD
$size = [math]::Round((Get-Item lowlight-code.zip).Length / 1KB)
Write-Host "Created lowlight-code.zip (commit $commit, $size KB)."
Write-Host "Upload it to: My Drive/lowlight-seg/code/lowlight-code.zip"
