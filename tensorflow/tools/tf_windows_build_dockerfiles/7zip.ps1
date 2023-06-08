# These lines consist of sub-commands that are executed using a single RUN command in the Dockerfile to install 7-Zip.
(New-Object Net.WebClient).DownloadFile("https://www.7-zip.org/a/7z2201-x64.msi", "7z.msi"); `
Start-Process -FilePath msiexec.exe -ArgumentList "/i 7z.msi /qn /norestart /log C:\tmp\7z_install_log.txt" -Wait; `
Remove-Item .\7z.msi; `
Write-Host "7z Installed.";
