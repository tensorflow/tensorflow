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
  [string]$Url = 'https://aka.ms/vs/17/release/vs_buildtools.exe',
  [string]$InstallPath = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
  [string []]$Components = @(
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    'Microsoft.VisualStudio.Component.Windows11SDK.26100',
    'Microsoft.VisualStudio.Workload.VCTools'
  )
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing Visual Studio 2022 Build Tools...'
$installerPath = Join-Path $env:TEMP 'vs_buildtools.exe'

Download-File -Url $Url -Destination $installerPath

$argsList = @('--installPath', $InstallPath, '--quiet', '--wait', '--norestart', '--nocache')
foreach ($comp in $Components) {
  $argsList += '--add'
  $argsList += $comp
}

Write-Output 'Running Visual Studio installer with components:'
$Components | ForEach-Object {
  Write-Output('  - {0}' -f $_)
}

# Packages and component versions can be found here:
# https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools
$process = Start-Process -FilePath $installerPath -ArgumentList $argsList -Wait -PassThru

# Check exit code: 3010 means success with reboot required, which is normally fine in containers
if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3010) {
  throw ('Visual Studio installer failed with exit code: {0}' -f $process.ExitCode)
}

Remove-Item -Path $installerPath -Force -ErrorAction SilentlyContinue
Write-Output 'Visual Studio 2022 Build Tools installation complete.'
