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
  [string]$Url = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.4/LLVM-18.1.4-win64.exe',
  [string]$TargetDir = 'C:\tools\LLVM'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing LLVM...'
$installerPath = Join-Path $env:TEMP 'LLVM.exe'
Download-File -Url $Url -Destination $installerPath

Write-Output('Extracting LLVM to {0}...' -f $TargetDir)
& 7z.exe 'x', $installerPath, "-o$TargetDir"

Write-Output 'Adding LLVM to Machine PATH...'
$binDir = Join-Path $TargetDir 'bin'
Add-ToMachinePath -Directory $binDir

Remove-Item -Path $installerPath -Force -ErrorAction SilentlyContinue
Write-Output 'LLVM installation complete.'
