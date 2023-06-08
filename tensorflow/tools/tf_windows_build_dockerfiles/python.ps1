$url = ('https://www.python.org/ftp/python/{0}/python-{0}-amd64.exe' -f $env:PYTHON_VERSION); `
Write-Host ('Downloading {0} ...' -f $url); `
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; `
(New-Object Net.WebClient).DownloadFile($url, 'C:\tmp\pyinstall.exe'); `
Write-Host 'Installing...'; `
   Start-Process -FilePath "C:\tmp\pyinstall.exe" -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "TargetDir=C:\Python" -Wait; `
Write-Host 'Removing ...'; `
Remove-Item C:\tmp\pyinstall.exe -Force; `
Write-Host 'Python Installed.';

