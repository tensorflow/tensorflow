# These lines consist of sub-commands that are executed using a single RUN command in the Dockerfile to install msys2, and add some extra tools.
(New-Object Net.WebClient).DownloadFile('https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20220603.tar.xz', 'msys2.tar.xz'); `
Start-Process -FilePath "C:\Program Files\7-Zip\7z.exe" -ArgumentList 'x msys2.tar.xz -oC:\tmp\msys2.tar' -Wait; `
Start-Process -FilePath "C:\Program Files\7-Zip\7z.exe" -ArgumentList 'x C:\tmp\msys2.tar -oC:\tools' -Wait; `
$env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';C:\tools\msys64;C:\tools\msys64\usr\bin\'; `
[Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine'); `
Write-Host "MSYS2 Installed.";
