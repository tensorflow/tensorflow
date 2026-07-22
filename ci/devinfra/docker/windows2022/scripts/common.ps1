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

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$ProgressPreference = 'SilentlyContinue'
$VerbosePreference = 'Continue'

<#
.SYNOPSIS
  Downloads a file from a URL to a specified local destination path with automatic retries,
  and optionally verifies its SHA256 checksum.
#>
function Download-File {
  param (
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $false)]
    [string]$Sha256 = ''
  )

  Write-Output ('Downloading from: {0}' -f $Url)
  Invoke-WebRequest -Uri $Url -OutFile $Destination -MaximumRetryCount 3 -RetryIntervalSec 5

  if ($Sha256) {
    Write-Output 'Verifying SHA256 hash...'
    $actualHash = (Get-FileHash -Path $Destination -Algorithm SHA256).Hash
    if ($actualHash -ne $Sha256.Trim()) {
      Remove-Item -Path $Destination -Force -ErrorAction SilentlyContinue
      throw ("SHA256 verification failed for '{0}'!`n  Expected: {1}`n  Actual:   {2}" -f $Destination, $Sha256, $actualHash)
    }
    Write-Output ('SHA256 verification passed: {0}' -f $actualHash)
  }
}

<#
.SYNOPSIS
  Appends a directory to the Machine-level PATH environment variable if not already present.
#>
function Add-ToMachinePath {
  param (
    [Parameter(Mandatory = $true)]
    [string]$Directory
  )

  $cleanDir = $Directory.TrimEnd('\')
  $machinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
  $machineParts = $machinePath.Split(';') | ForEach-Object { $_.TrimEnd('\') }
  if ($machineParts -notcontains $cleanDir) {
    $newPath = $machinePath.TrimEnd(';') + ';' + $Directory
    [Environment]::SetEnvironmentVariable('PATH', $newPath, 'Machine')
  }

  $envParts = $env:PATH.Split(';') | ForEach-Object { $_.TrimEnd('\') }
  if ($envParts -notcontains $cleanDir) {
    $env:PATH = $env:PATH.TrimEnd(';') + ';' + $Directory
  }
}
