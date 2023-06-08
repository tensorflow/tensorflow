# These lines consist of sub-commands that are executed using a single RUN command in the Dockerfile to install jdk by using zulu.
Add-Type -AssemblyName "System.IO.Compression.FileSystem"; `
$zulu_url = "https://cdn.azul.com/zulu/bin/zulu8.28.0.1-jdk8.0.163-win_x64.zip"; `
$zulu_zip = "c:\tmp\zulu8.28.0.1-jdk8.0.163-win_x64.zip"; `
$zulu_extracted_path = "c:\tmp\" + [IO.Path]::GetFileNameWithoutExtension($zulu_zip); `
$zulu_root = "c:\openjdk"; `
(New-Object Net.WebClient).DownloadFile($zulu_url, $zulu_zip); `
[System.IO.Compression.ZipFile]::ExtractToDirectory($zulu_zip, "c:\tmp"); `
Move-Item $zulu_extracted_path -Destination $zulu_root; `
Remove-Item $zulu_zip; `
$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";${zulu_root}\bin"; `
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, "Machine"); `
$env:JAVA_HOME = $zulu_root; `
[Environment]::SetEnvironmentVariable("JAVA_HOME", $env:JAVA_HOME, "Machine"); `
Write-Host "jdk Installed.";
