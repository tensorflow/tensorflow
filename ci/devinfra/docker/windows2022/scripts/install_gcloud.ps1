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
  [string]$Url = 'https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing Google Cloud SDK...'
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = 1

$zipPath = Join-Path $env:TEMP 'google-cloud-sdk.zip'
Download-File -Url $Url -Destination $zipPath

Write-Output('Extracting SDK to {0}...' -f $env:ProgramFiles)
Expand-Archive -Path $zipPath -DestinationPath $env:ProgramFiles -Force

$installBat = Join-Path $env:ProgramFiles 'google-cloud-sdk\install.bat'
Write-Output 'Running SDK install.bat...'
& $installBat --path-update false

$binDir = Join-Path $env:ProgramFiles 'google-cloud-sdk\bin'
Write-Output 'Adding Google Cloud SDK bin to Machine PATH...'
Add-ToMachinePath -Directory $binDir

$msysRoot = 'C:\tools\msys64'
if (Test-Path $msysRoot) {
  $msysBashrc = Join-Path $msysRoot '.bashrc'
  # MSYS bash installed by
  # //learning/brain/testing/ml_oss/ml_velocity/container/tf_test_windows/scripts/install_msys2.ps1
  # attempts to execute extensionless binaries rather than requiring `.cmd`, which leads to
  # path resolution issues.
  # These aliases ensure bash properly invokes `gcloud.cmd`, `gsutil.cmd`, and `bq.cmd`.
  Write-Output 'Adding Google Cloud SDK aliases to MSYS .bashrc...'
  Add-Content -Path $msysBashrc -Value 'alias gcloud=gcloud.cmd'
  Add-Content -Path $msysBashrc -Value 'alias gsutil=gsutil.cmd'
  Add-Content -Path $msysBashrc -Value 'alias bq=bq.cmd'
}

Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
Write-Output 'Google Cloud SDK installation complete.'
