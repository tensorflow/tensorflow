# This Dockerfile creates an image that has:
# - the correct MTU setting for networking from inside the container to work.
# - Visual Studio 2022 Build Tools
# - MSVC 14.39
# - LLVM/Clang 18.1.4
# - MSYS2 + curl, git, patch, vim, unzip, zip
# - Python 3.9 - 3.13
# - Bazelisk 1.22.1
# - JDK 21 (Azul Zulu)

FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", \
       "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';$VerbosePreference = 'Continue';"]

# Enable long paths
RUN New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

RUN md C:\TEMP
RUN md C:\TMP
ENV TMP "C:/TMP"
ENV TEMP "C:/TEMP"

# Install 7-Zip.
RUN (New-Object Net.WebClient).DownloadFile('https://www.7-zip.org/a/7z2201-x64.msi', '7z.msi'); \
    Start-Process msiexec.exe -ArgumentList \"/i 7z.msi /qn /norestart /log C:\\TEMP\\7z_install_log.txt\" -wait; \
    Remove-Item .\7z.msi;

# Download the Visual Studio 2022 Installer.
RUN (New-Object Net.WebClient).DownloadFile('https://aka.ms/vs/17/release/vs_community.exe', 'C:\TEMP\vs_community.exe');
# Install Visual Studio 2022 Build Tools + Compiler
SHELL ["cmd", "/S", "/C"]
# Packages, and component versions, can be found here:
# https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools
RUN C:\TEMP\vs_community.exe \
        --quiet --wait --norestart --nocache \
        --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        --add Microsoft.VisualStudio.Workload.NativeDesktop \
        --add Microsoft.VisualStudio.Component.VC.14.39.17.9.x86.64 \
        --add Microsoft.VisualStudio.Component.Windows11SDK.22621 \
 || IF "%ERRORLEVEL%"=="3010" EXIT 0

SHELL ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", \
       "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $VerbosePreference = 'Continue';"]

# Install Clang.
RUN (New-Object Net.WebClient).DownloadFile( \
         'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.4/LLVM-18.1.4-win64.exe', \
         'LLVM.exe'); \
    Start-Process -FilePath \"C:\Program Files\7-Zip\7z.exe\" -ArgumentList 'x LLVM.exe -oC:\tools\LLVM' -Wait; \
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';C:\tools\LLVM\bin'; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine');

# Install MSYS2.
RUN (New-Object Net.WebClient).DownloadFile( \
         'https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20240727.tar.xz', \
         'msys2.tar.xz'); \
    Start-Process -FilePath \"C:\Program Files\7-Zip\7z.exe\" -ArgumentList 'x msys2.tar.xz -oC:\TEMP\msys2.tar' -Wait; \
    Start-Process -FilePath \"C:\Program Files\7-Zip\7z.exe\" -ArgumentList 'x C:\TEMP\msys2.tar -oC:\tools' -Wait; \
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';C:\tools\msys64;C:\tools\msys64\usr\bin\'; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine');

# Disable signature checking on pacman because we cannot initialize the keyring.
RUN Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.mingw32 -Value 'SigLevel = Never'
RUN Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.mingw64 -Value 'SigLevel = Never'
RUN Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.msys -Value 'SigLevel = Never'

# Install pacman packages.
RUN C:\tools\msys64\usr\bin\bash.exe -lc \
  'pacman --noconfirm -Syy curl git patch vim unzip zip'

# Install multiple Pythons, but only add one of them to PATH.
RUN function Install-Python { \
        param( \
            [string]$version, \
            [int]$prependPath \
        ) \
        $url = ('https://www.python.org/ftp/python/{0}/python-{0}-amd64.exe' -f $version); \
        Write-Host ('Downloading {0}...' -f $url); \
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
        (New-Object Net.WebClient).DownloadFile($url, 'C:\tmp\pyinstall.exe'); \
        \
        # Without the patch version \
        $truncatedVersion = $($version -replace '\.\d+$', ''); \
        $installDir = 'C:\Python' + $truncatedVersion; \
        Write-Host ('Installing into {0} (PrependPath: {1})...' -f $installDir, $($prependPath -eq 1)); \
        $argumentList = ('/quiet InstallAllUsers=1 PrependPath={0} TargetDir={1}' -f $prependPath, $installDir); \
        Start-Process -FilePath 'C:\tmp\pyinstall.exe' -ArgumentList $argumentList -Wait; \
        \
        Write-Host 'Verifying install...'; \
        Write-Host \"  python --version $version\"; & $installDir\python.exe --version; \
        \
        Write-Host 'Verifying pip install...'; \
        & $installDir\python.exe -m pip --version; \
        \
        Write-Host 'Updating pip...'; \
        & $installDir\python.exe -m pip install --upgrade pip; \
        \
        Write-Host 'Installing/updating packages...'; \
        & $installDir\python.exe -m pip install --upgrade setuptools packaging; \
        \
        Write-Host 'Removing installation binary...'; \
        Remove-Item C:\tmp\pyinstall.exe -Force; \
    }; \
    Write-Host 'Installing multiple Python versions...'; \
    $versions = @( \
        @{ version = '3.9.13'; prependPath = 0 }, \
        @{ version = '3.10.11'; prependPath = 0 }, \
        @{ version = '3.11.9'; prependPath = 0 }, \
        @{ version = '3.12.8'; prependPath = 0 }, \
        @{ version = '3.13.1'; prependPath = 1 } \
    ); \
    foreach ($v in $versions) { \
        Install-Python -version $v.version -prependPath $v.prependPath; \
    }; \
    Write-Host 'Python installations complete.';

# Add a python3 symlink for the Python in PATH.
# It's not feasible to add one for each version, as on Windows,
# Python uses PATH and the binary's (symlink's) location, to launch itself.
RUN C:\tools\msys64\usr\bin\bash.exe -lc 'ln -s /c/Python3.13/python.exe /usr/bin/python3';

# Install JDK 21.
RUN \
  Add-Type -AssemblyName \"System.IO.Compression.FileSystem\"; \
  $zulu_pkg = \"zulu21.34.19-ca-jdk21.0.3-win_x64.zip\"; \
  $zulu_url = \"https://cdn.azul.com/zulu/bin/${zulu_pkg}\"; \
  $zulu_zip = \"c:\\temp\\${zulu_pkg}\"; \
  $zulu_extracted_path = \"c:\\temp\\\" + [IO.Path]::GetFileNameWithoutExtension($zulu_zip); \
  $zulu_root = \"c:\\openjdk\"; \
  (New-Object Net.WebClient).DownloadFile($zulu_url, $zulu_zip); \
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zulu_zip, \"c:\\temp\"); \
  Move-Item $zulu_extracted_path -Destination $zulu_root; \
  Remove-Item $zulu_zip; \
  $env:PATH = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\") + \";${zulu_root}\\bin\"; \
  [Environment]::SetEnvironmentVariable(\"PATH\", $env:PATH, \"Machine\"); \
  $env:JAVA_HOME = $zulu_root; \
  [Environment]::SetEnvironmentVariable(\"JAVA_HOME\", $env:JAVA_HOME, \"Machine\")

# Point to the LLVM installation.
# The Bazel Windows guide claims it can find LLVM automatically,
# but it likely only works if it's installed somewhere inside C:\Program Files.
ENV BAZEL_LLVM "C:\tools\LLVM"

# These variables may be useful, but so far haven't been. Keeping for posterity.
# ENV CLANG_COMPILER_PATH "C:\tools\llvm\bin\clang.exe"
# ENV CC "C:\tools\llvm\bin\clang.exe"
# ENV BAZEL_COMPILER "C:\tools\llvm\bin\clang.exe"

ENV BAZEL_SH "C:\tools\msys64\usr\bin\bash.exe"
ENV BAZEL_VS "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
ENV BAZEL_VC "C:\Program Files\Microsoft Visual Studio\2022\Community\VC"

# Environment variables to prevent auto-conversion of Linux-like paths to Windows paths.
# This is necessary as some paths end up invalid/mangled.
ENV MSYS_NO_PATHCONV 1
ENV MSYS2_ARG_CONV_EXCL *

# This should only be necessary if there are multiple, differently-versioned
# MSVC compilers installed, and a particular one should be used.
# To find exact versions available:
# - Navigate to the relevant folder, e.g.
#     C:\Program Files\Microsoft Visual Studio\2022
# - Search for the `cl.exe` file: `gci -r -fi cl.exe`
# - The version will be part of the found path, e.g.
#     2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64
# ENV BAZEL_VC_FULL_VERSION 14.39.33519

# Install Bazelisk.
RUN md C:\tools\bazel
RUN (New-Object Net.WebClient).DownloadFile( \
         'https://github.com/bazelbuild/bazelisk/releases/download/v1.22.1/bazelisk-windows-amd64.exe', \
         'C:\tools\bazel\bazel.exe'); \
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';C:\tools\bazel'; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine');

# Install gcloud, and add it to PATH
ENV CLOUDSDK_CORE_DISABLE_PROMPTS 1
RUN (New-Object Net.WebClient).DownloadFile('https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.zip', 'C:\Temp\google-cloud-sdk.zip'); \
    Expand-Archive -Path 'C:\Temp\google-cloud-sdk.zip' -DestinationPath $env:ProgramFiles -Verbose:$false
RUN & \"$env:ProgramFiles\\google-cloud-sdk\\install.bat\" --path-update false
RUN $env:Path += \";$env:ProgramFiles\\google-cloud-sdk\\bin\"; \
    [Environment]::SetEnvironmentVariable('Path', $env:Path, [EnvironmentVariableTarget]::Machine);
# Re-enable prompts for interactive use.
ENV CLOUDSDK_CORE_DISABLE_PROMPTS=""

# MSYS attempts to use non-cmd versions, which aren't meant for Windows
RUN Add-Content -Path C:\tools\msys64\.bashrc -Value 'alias gcloud=gcloud.cmd'
RUN Add-Content -Path C:\tools\msys64\.bashrc -Value 'alias gsutil=gsutil.cmd'
RUN Add-Content -Path C:\tools\msys64\.bashrc -Value 'alias bq=bq.cmd'

# Symlink a directory, to have it pretend be the T:\ drive.
# This drive letter is used by internal CI,
# and part of it is mounted to the container during the container's creation.
#
# While the mount argument (`-v host_path:container_path`) still requires
# `container_path` to be a legitimate C:\ path, in this case, 'C:\drive_t',
# this symlink does allow for the convenience of passing unedited paths
# to `docker exec` commands, e.g., 'T:\path', instead of 'C:\path',
# without having to replace the drive letter with C:\ every time.
# Such a workaround is not required on Linux, since it
# can create arbitrary paths within the container, e.g., '/t'.
# Note: This does not affect/work for `docker cp` commands.
RUN New-Item -ItemType directory -Path C:\drive_t; \
    New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\DOS Devices' -Name 'T:'  -Value '\??\C:\drive_t' -PropertyType String;

CMD ["bash", "-i"]
