# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

param (
  # Refer to https://repo.msys2.org/distrib/x86_64.
  [string]$Url = 'https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20260611.tar.xz',
  [string]$TargetDir = 'C:\tools',
  [string []]$Packages = @('curl', 'git', 'patch', 'vim', 'unzip', 'wget', 'zip')
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing MSYS2...'
$txzPath = Join-Path $env:TEMP 'msys2.tar.xz'
$tarDir = Join-Path $env:TEMP 'msys2_tar'
New-Item -ItemType Directory -Path $tarDir -Force | Out-Null
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

Download-File -Url $Url -Destination $txzPath

Write-Output 'Extracting .tar.xz archive...'
& 7z.exe 'x', $txzPath, "-o$tarDir"

$tarPath = Get-ChildItem -Path $tarDir -Filter '*.tar' | Select-Object -ExpandProperty FullName -First 1
Write-Output('Extracting .tar archive to {0}...' -f $TargetDir)
& 7z.exe 'x', $tarPath, "-o$TargetDir"

$msysRoot = Join-Path $TargetDir 'msys64'
$msysBin = Join-Path $msysRoot 'usr\bin'

Write-Output 'Adding MSYS2 to Machine PATH...'
Add-ToMachinePath -Directory $msysRoot
Add-ToMachinePath -Directory $msysBin

# Disable signature checking on pacman because we cannot initialize the keyring in container builds.
Write-Output 'Configuring pacman.conf...'
$pacmanConf = Join-Path $msysRoot 'etc\pacman.conf'
if (Test-Path $pacmanConf) {
  (Get-Content $pacmanConf) -replace '^SigLevel\s*=.*', 'SigLevel = Never' | Set-Content $pacmanConf
}

# Install pacman packages
if ($Packages -and $Packages.Count -gt 0) {
  Write-Output 'Installing pacman packages...'
  $bashExe = Join-Path $msysBin 'bash.exe'
  $pkgList = $Packages -join ' '
  & $bashExe '-lc', "pacman --noconfirm -Syy $pkgList"
}

Remove-Item -Path $txzPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $tarDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Output 'MSYS2 installation complete.'
