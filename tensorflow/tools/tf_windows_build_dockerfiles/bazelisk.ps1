# These lines consist of sub-commands that are executed using a single RUN command in the Dockerfile to install Bazelisk. 
(New-Object Net.WebClient).DownloadFile( `
     "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-windows-amd64.exe", `
     "C:\tools\bazel\bazel.exe"); `
$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";C:\tools\bazel"; `
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, "Machine"); `
Write-Host "bazelisk Installed.";
