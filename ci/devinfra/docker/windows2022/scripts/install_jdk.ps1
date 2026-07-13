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
  # For reference: https://www.azul.com/downloads/?package=jdk#zulu.
  [string]$PackageName = 'zulu25.34.17-ca-jdk25.0.3-win_x64.zip',
  [string]$BaseUrl = 'https://cdn.azul.com/zulu/bin',
  [string]$TargetDir = 'C:\openjdk'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing JDK...'
Add-Type -AssemblyName 'System.IO.Compression.FileSystem'

$url = ('{0}/{1}' -f $BaseUrl, $PackageName)
$zipPath = Join-Path $env:TEMP $PackageName
$extractedName = [IO.Path]::GetFileNameWithoutExtension($PackageName)
$extractedPath = Join-Path $env:TEMP $extractedName

Download-File -Url $url -Destination $zipPath

Write-Output 'Extracting JDK zip directly to target location...'
$parentDir = Split-Path $TargetDir -Parent
$targetLeaf = Split-Path $TargetDir -Leaf

[System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $parentDir)
$extractedPath = Join-Path $parentDir $extractedName

Write-Output('Renaming extracted JDK to {0}...' -f $targetLeaf)
Rename-Item -Path $extractedPath -NewName $targetLeaf

Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue

Write-Output 'Setting JAVA_HOME and updating Machine PATH...'
$binDir = Join-Path $TargetDir 'bin'
Add-ToMachinePath -Directory $binDir

[Environment]::SetEnvironmentVariable('JAVA_HOME', $TargetDir, 'Machine')
Write-Output 'JDK installation complete.'
