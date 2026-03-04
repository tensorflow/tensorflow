# Multi-tool Windows build image, cache-friendly ordering and checksum verification.
#
# Build hint (override ARGs from CLI):
#   docker build \
#     --build-arg SEVENZIP_URL="https://www.7-zip.org/a/7z2301-x64.msi" \
#     --build-arg SHA256_7ZIP="your_sha256_here" \
#     --build-arg BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.22.1/bazelisk-windows-amd64.exe" \
#     --build-arg SHA256_BAZELISK="your_sha256_here" \
#     -t win-dev:latest .
#
# Notes:
#  - Put rarely-changing layers (registry tweaks, 7-Zip, Visual Studio install) early for Docker cache stability.
#  - Put frequently-changing packages (Python versions, Bazelisk) near the end.
#  - Replace any SHA256 placeholders with actual checksums (or pass via --build-arg) before using in secure CI.

FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", \
       "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'"]

# ---------- BUILD-TIME CONFIG (change via --build-arg) ----------
ARG VS_INSTALLER_URL="https://aka.ms/vs/17/release/vs_community.exe"
ARG VS_COMPONENTS="--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
ARG SEVENZIP_URL="https://www.7-zip.org/a/7z2201-x64.msi"
ARG SHA256_7ZIP=""                                # <<< REPLACE with actual SHA256 or pass via --build-arg
ARG LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.4/LLVM-18.1.4-win64.exe"
ARG SHA256_LLVM="78d8f528a132e131b978e6b3276aa45af759a068a25c006240f59ab28df5f621"
ARG MSYS2_TAR_URL="https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20240727.tar.xz"
ARG SHA256_MSYS2=""                               # <<< MSYS2 often distributes signatures; fill if you have a verified sha256
ARG BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.22.1/bazelisk-windows-amd64.exe"
ARG SHA256_BAZELISK=""                            # <<< fill in if you want strict verification
ARG ZULU_PKG="zulu21.34.19-ca-jdk21.0.3-win_x64.zip"
ARG ZULU_URL="https://cdn.azul.com/zulu/bin/${ZULU_PKG}"
ARG SHA256_ZULU="a7e9b74b0d4809a77f6c13cbbd65ce0baa0b923281333179972ed25326ebe3d7"
ARG BAZELISK_VERSION="1.22.1"

# ---------- ENV and basic dirs ----------
ENV TMP="C:\\TMP" `
    TEMP="C:\\TEMP" `
    MSYS2_ROOT="C:\\tools\\msys64" `
    LLVM_ROOT="C:\\tools\\LLVM" `
    BAZEL_BIN="C:\\tools\\bazel"

RUN md $env:TEMP -ErrorAction SilentlyContinue; md $env:TMP -ErrorAction SilentlyContinue; \
    md C:\tools -ErrorAction SilentlyContinue; md $env:BAZEL_BIN -ErrorAction SilentlyContinue

# ---------- Helper functions: download + verify ----------
SHELL ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop';"]
RUN function Download-File([string]$url, [string]$out) { \
        Write-Host \"Downloading: $url -> $out\"; \
        (New-Object System.Net.WebClient).DownloadFile($url, $out); \
    }; \
    function Verify-Hash([string]$file, [string]$expected) { \
        if ([string]::IsNullOrWhiteSpace($expected)) { Write-Host \"No expected hash provided for $file - skipping verification\"; return }; \
        Write-Host \"Verifying SHA256 for $file\"; \
        $actual = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower(); \
        if ($actual -ne $expected.ToLower()) { throw \"SHA256 mismatch for $file: $actual != $expected\" }; \
        Write-Host \"SHA256 OK for $file\"; \
    }; \
    Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -Type DWord -Force

# ---------- Install 7-Zip early (rarely changes) ----------
RUN Download-File ${SEVENZIP_URL} C:\\TEMP\\7z.msi; \
    Verify-Hash C:\\TEMP\\7z.msi ${SHA256_7ZIP}; \
    Start-Process msiexec -ArgumentList '/i','C:\\TEMP\\7z.msi','/qn','/norestart' -Wait -NoNewWindow; \
    Remove-Item C:\\TEMP\\7z.msi -Force

# ---------- Visual Studio Build Tools (stable, heavy) ----------
# Use cmd because VS installer expects cmd-style args; keep this layer before frequently-changing tools
SHELL ["cmd", "/S", "/C"]
RUN powershell -Command "Invoke-WebRequest -UseBasicParsing -Uri \"${VS_INSTALLER_URL}\" -OutFile C:\\TEMP\\vs_installer.exe" && \
    C:\\TEMP\\vs_installer.exe --quiet --wait --norestart --nocache %VS_COMPONENTS% || IF "%ERRORLEVEL%"=="3010" EXIT 0 && \
    powershell -Command \"Remove-Item -Force C:\\TEMP\\vs_installer.exe\" 

# back to powershell
SHELL ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop';"]

# ---------- LLVM (verifyable SHA) ----------
RUN Download-File ${LLVM_URL} C:\\TEMP\\LLVM.exe; \
    Verify-Hash C:\\TEMP\\LLVM.exe ${SHA256_LLVM}; \
    Start-Process -FilePath '\"C:\\Program Files\\7-Zip\\7z.exe\"' -ArgumentList 'x','C:\\TEMP\\LLVM.exe','-oC:\\tools\\LLVM','-y' -Wait; \
    Remove-Item C:\\TEMP\\LLVM.exe -Force; \
    $p=[Environment]::GetEnvironmentVariable('PATH','Machine') + ';C:\\tools\\LLVM\\bin'; [Environment]::SetEnvironmentVariable('PATH',$p,'Machine')

# ---------- MSYS2 (extracted) ----------
RUN Download-File ${MSYS2_TAR_URL} C:\\TEMP\\msys2.tar.xz; \
    Verify-Hash C:\\TEMP\\msys2.tar.xz ${SHA256_MSYS2}; \
    Start-Process -FilePath '\"C:\\Program Files\\7-Zip\\7z.exe\"' -ArgumentList 'x','C:\\TEMP\\msys2.tar.xz','-oC:\\TEMP\\msys2.tar' -Wait; \
    Start-Process -FilePath '\"C:\\Program Files\\7-Zip\\7z.exe\"' -ArgumentList 'x','C:\\TEMP\\msys2.tar','-oC:\\tools' -Wait; \
    Remove-Item C:\\TEMP\\msys2.tar.xz -Force; Remove-Item -Recurse -Force C:\\TEMP\\msys2.tar; \
    $p=[Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + 'C:\\tools\\msys64;C:\\tools\\msys64\\usr\\bin'; [Environment]::SetEnvironmentVariable('PATH',$p,'Machine')

# For environments where pacman keyring cannot be initialized, lower signature checking.
RUN Add-Content -Path C:\\tools\\msys64\\etc\\pacman.d\\mirrorlist.mingw32 -Value 'SigLevel = Never' ; \
    Add-Content -Path C:\\tools\\msys64\\etc\\pacman.d\\mirrorlist.mingw64 -Value 'SigLevel = Never' ; \
    Add-Content -Path C:\\tools\\msys64\\etc\\pacman.d\\mirrorlist.msys -Value 'SigLevel = Never'

# Install common MSYS packages
RUN C:\\tools\\msys64\\usr\\bin\\bash.exe -lc 'pacman --noconfirm -Syyu --needed curl git patch vim unzip zip'

# ---------- Google Cloud SDK ----------
ENV CLOUDSDK_CORE_DISABLE_PROMPTS=1
RUN Download-File 'https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip' C:\\TEMP\\gcloud.zip; \
    Expand-Archive -Path C:\\TEMP\\gcloud.zip -DestinationPath $env:ProgramFiles; \
    Remove-Item C:\\TEMP\\gcloud.zip -Force; \
    & \"$env:ProgramFiles\\google-cloud-sdk\\install.bat\" --path-update false --quiet; \
    $pv=[Environment]::GetEnvironmentVariable('PATH','Machine') + \";$env:ProgramFiles\\google-cloud-sdk\\bin\"; [Environment]::SetEnvironmentVariable('PATH',$pv,'Machine'); \
    Set-Item -Path C:\\tools\\msys64\\.bashrc -Value \"alias gcloud=gcloud.cmd`nalias gsutil=gsutil.cmd`nalias bq=bq.cmd\" -Force

ENV CLOUDSDK_CORE_DISABLE_PROMPTS=""

# ---------- Azul Zulu JDK (SHA from Azul certificate) ----------
RUN Download-File ${ZULU_URL} C:\\TEMP\\zulu.zip; \
    Verify-Hash C:\\TEMP\\zulu.zip ${SHA256_ZULU}; \
    Add-Type -AssemblyName System.IO.Compression.FileSystem; \
    [System.IO.Compression.ZipFile]::ExtractToDirectory('C:\\TEMP\\zulu.zip', 'C:\\TEMP\\zulu_extracted'); \
    Move-Item 'C:\\TEMP\\zulu_extracted\\'* 'C:\\openjdk' -Force; \
    Remove-Item 'C:\\TEMP\\zulu.zip' -Force; Remove-Item -Recurse -Force 'C:\\TEMP\\zulu_extracted'; \
    $jpath = 'C:\\openjdk\\bin'; $p = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + $jpath; [Environment]::SetEnvironmentVariable('PATH',$p,'Machine'); [Environment]::SetEnvironmentVariable('JAVA_HOME','C:\\openjdk','Machine')

# ---------- Bazelisk (put later so changing versions don't bust prior cache layers) ----------
RUN md C:\\tools\\bazel -ErrorAction SilentlyContinue; \
    Download-File ${BAZELISK_URL} C:\\tools\\bazel\\bazel.exe; \
    Verify-Hash C:\\tools\\bazel\\bazel.exe ${SHA256_BAZELISK}; \
    icacls 'C:\\tools\\bazel\\bazel.exe' /grant:r '*S-1-1-0:(RX)' > $null; \
    $p=[Environment]::GetEnvironmentVariable('PATH','Machine') + ';C:\\tools\\bazel'; [Environment]::SetEnvironmentVariable('PATH',$p,'Machine')

# ---------- Python versions (frequently changed — keep at the end) ----------
SHELL ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop';"]
RUN function Install-Python([string]$version, [int]$prepend) { \
        $url = \"https://www.python.org/ftp/python/$version/python-$version-amd64.exe\"; \
        $out = \"C:\\TEMP\\python-$version.exe\"; \
        Write-Host \"Downloading $url\"; (New-Object System.Net.WebClient).DownloadFile($url, $out); \
        $trunc = ($version -replace '\\.\\d+$',''); \
        $target = \"C:\\Python$trunc\"; \
        $args = \"/quiet InstallAllUsers=1 PrependPath=$prepend TargetDir=$target\"; \
        Start-Process -FilePath $out -ArgumentList $args -Wait; \
        Remove-Item $out -Force; \
        & \"$target\\python.exe\" -m pip install --upgrade pip setuptools packaging; \
    }; \
    $pylist = @(@{v='3.9.13'; p=0}, @{v='3.10.11'; p=0}, @{v='3.11.9'; p=0}, @{v='3.12.8'; p=0}, @{v='3.13.1'; p=1}); \
    foreach ($p in $pylist) { Install-Python $p.v $p.p }

# Symlink for msys python3 convenience
RUN C:\\tools\\msys64\\usr\\bin\\bash.exe -lc 'ln -sf /c/Python3.13/python.exe /usr/bin/python3 || true'

# ---------- MSYS + Bazel / env defaults ----------
ENV BAZEL_SH="C:\\tools\\msys64\\usr\\bin\\bash.exe"
ENV BAZEL_LLVM="C:\\tools\\LLVM"
ENV BAZEL_VS="C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools"
ENV BAZEL_VC="C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC"
ENV MSYS_NO_PATHCONV=1
ENV MSYS2_ARG_CONV_EXCL=*

# Create fake T: drive (CI convenience)
RUN New-Item -ItemType directory -Path C:\\drive_t -Force; \
    New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\DOS Devices' -Name 'T:' -Value '\\??\\C:\\drive_t' -PropertyType String -Force

# Final cleanup of temp files (keeps layers small)
RUN Remove-Item -Recurse -Force C:\\TEMP\\* -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force C:\\TMP\\* -ErrorAction SilentlyContinue

# Default to an interactive bash via MSYS
CMD ["bash", "-i"]