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
  [Parameter(Mandatory = $true)] [string]$Version,
  [Parameter(Mandatory = $true)] [string]$TargetDir,
  [string]$FtpDir = '',
  [string []]$PipPackages = @('setuptools', 'packaging'),
  [string []]$Symlinks = @()
)

. "$PSScriptRoot\common.ps1"

if (-not $FtpDir) {
  $FtpDir = $Version
}

Write-Output('Installing Python {0} to {1}...' -f $Version, $TargetDir)

$url = ('https://www.python.org/ftp/python/{0}/python-{1}-amd64.exe' -f $FtpDir, $Version)
$installerPath = Join-Path $env:TEMP 'pyinstall.exe'

Download-File -Url $url -Destination $installerPath

Write-Output 'Running Python installer...'
$argsList = ('/quiet InstallAllUsers=1 PrependPath=1 TargetDir="{0}"' -f $TargetDir)
$process = Start-Process -FilePath $installerPath -ArgumentList $argsList -Wait -PassThru

if ($process.ExitCode -ne 0) {
  throw ('Python installer failed with exit code {0}' -f $process.ExitCode)
}

$pythonExe = Join-Path $TargetDir 'python.exe'
if (-not (Test-Path $pythonExe)) {
  throw ('Python installation failed; {0} not found.' -f $pythonExe)
}

Write-Output 'Verifying install:'
& $pythonExe --version
& $pythonExe -m pip --version

Remove-Item -Path $installerPath -Force -ErrorAction SilentlyContinue

if ($PipPackages -and $PipPackages.Count -gt 0) {
  Write-Output 'Installing / upgrading pip packages:'
  $PipPackages | ForEach-Object {
    Write-Output('  - {0}' -f $_)
  }
  $pipArgs = @('-m', 'pip', 'install', '--ignore-installed', '--upgrade') + $PipPackages
  & $pythonExe $pipArgs
}

if ($Symlinks -and $Symlinks.Count -gt 0) {
  Write-Output 'Creating symbolic links...'
  foreach ($linkName in $Symlinks) {
    $linkPath = Join-Path $TargetDir $linkName
    if (-not (Test-Path $linkPath)) {
      Write-Output('  Creating symlink: {0} -> {1}' -f $linkPath, $pythonExe)
      try {
        New-Item -ItemType SymbolicLink -Path $linkPath -Target $pythonExe -Force -ErrorAction Stop | Out-Null
      }
      catch {
        # Fallback to cmd mklink if required by permissions or filesystem
        cmd.exe /c ("mklink `"{0}`" `"{1}`"" -f $linkPath, $pythonExe) | Out-Null
        if ($LASTEXITCODE -ne 0) {
          throw ('Failed to create symlink using mklink with exit code {0}' -f $LASTEXITCODE)
        }
      }
    }
  }
}

Write-Output('Python {0} setup complete.' -f $Version)
