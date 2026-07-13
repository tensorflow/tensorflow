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
  [string]$Url = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip',
  [string]$TargetDir = 'C:\tools\uv'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing uv (standalone)...'
$zipPath = Join-Path $env:TEMP 'uv.zip'
Download-File -Url $Url -Destination $zipPath

Write-Output ('Extracting uv to {0}...' -f $TargetDir)
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
& 7z.exe 'e', $zipPath, "-o$TargetDir", 'uv-x86_64-pc-windows-msvc\*.exe'

Write-Output 'Adding uv directory to Machine PATH...'
Add-ToMachinePath -Directory $TargetDir

Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
Write-Output 'uv standalone installation complete.'
