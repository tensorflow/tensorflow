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
  # Refer to https://www.7-zip.org/download.html for available versions.
  [string]$Url = 'https://github.com/ip7z/7zip/releases/download/26.02/7z2602-x64.msi'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing 7-Zip...'
$msiPath = Join-Path $env:TEMP '7z.msi'
$logPath = Join-Path $env:TEMP '7z_install_log.txt'

Download-File -Url $Url -Destination $msiPath

Write-Output 'Running MSI installer...'
$process = Start-Process -FilePath 'msiexec.exe' -ArgumentList ('/i "{0}" /qn /norestart /log "{1}"' -f $msiPath, $logPath) -Wait -PassThru

if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3010) {
  if (Test-Path $logPath) {
    Get-Content $logPath | Select-Object -Last 20 | Write-Output
  }
  throw ('7-Zip installer failed with exit code: {0}' -f $process.ExitCode)
}

Remove-Item -Path $msiPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $logPath -Force -ErrorAction SilentlyContinue

Write-Output 'Adding 7-Zip directory to Machine PATH...'
Add-ToMachinePath -Directory (Join-Path $env:ProgramFiles '7-Zip')

Write-Output '7-Zip installation complete.'
