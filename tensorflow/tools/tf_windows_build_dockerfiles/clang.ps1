(New-Object Net.WebClient).DownloadFile("https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win64.exe", "LLVM.exe"); `
Start-Process -FilePath "C:\Program Files\7-Zip\7z.exe" -ArgumentList "x LLVM.exe -oC:\tools\llvm" -Wait; `
$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";C:\tools\llvm\bin"; `
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, "Machine"); `
Write-Host "clang Installed.";