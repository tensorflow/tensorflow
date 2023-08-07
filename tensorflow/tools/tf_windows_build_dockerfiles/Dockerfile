# This Dockerfile creates an image that:
# - Has the correct MTU setting for networking from inside the container to work.
# - Has Visual Studio 2019 Build Tools installed.
# - Has MSVC 14.29 and clang 14.0 installed
# - Has msys2 + git, curl, zip, unzip installed.
# - Has Python installation based on the input argument.
# - Has Bazelisk v1.16.0 installed.
FROM mcr.microsoft.com/windows/servercore:ltsc2019
SHELL ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $VerbosePreference = 'Continue';"]
# RUN netsh interface ipv4 set subinterface \"vEthernet (Ethernet)\" mtu=1460 store=persistent

RUN md C:\TMP

# Install Visual Studio 2019 Build Tools.
RUN (New-Object Net.WebClient).DownloadFile('https://aka.ms/vs/16/release/vs_community.exe', 'C:\tmp\vs_community.exe');
COPY vs.bat .
SHELL ["cmd", "/S", "/C"]
RUN vs.bat || IF "%ERRORLEVEL%"=="3010" EXIT 0

SHELL ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $VerbosePreference = 'Continue';"]
# Install 7-Zip and add it to the path.
COPY 7zip.ps1 7zip.ps1
RUN powershell -ExecutionPolicy Bypass -File 7zip.ps1 

# Install msys2, and add some extra tools.
COPY msys2.ps1 msys2.ps1
RUN powershell -ExecutionPolicy Bypass -File msys2.ps1 

# Install clang.
COPY clang.ps1 clang.ps1
RUN powershell -ExecutionPolicy Bypass -File clang.ps1 

# Enable signature checking on pacman and Install pacman packages.
COPY pacman.ps1 pacman.ps1
RUN powershell -ExecutionPolicy Bypass -File pacman.ps1 

# Install Python (specified by ARG PYTHON_VERSION)
ARG PYTHON_VERSION
COPY python.ps1 python.ps1
RUN powershell -ExecutionPolicy Bypass -File python.ps1 
RUN Write-Host '  python --version';& "python.exe" --version;

# Install the pip packages
COPY devel.requirements.txt C:\\tmp\\devel.requirements.txt
RUN & "python.exe" -m pip install --ignore-installed --force-reinstall --upgrade -r 'C:\tmp\devel.requirements.txt'

# Install jdk by using zulu
COPY jdk.ps1 jdk.ps1
RUN powershell -ExecutionPolicy Bypass -File jdk.ps1 

# Environment variables to work around the msys issues
ENV MSYS_NO_PATHCONV 1
ENV MSYS2_ARG_CONV_EXCL *

# See https://docs.microsoft.com/en-us/visualstudio/releases/2019/history for the mapping
ENV BAZEL_VC_FULL_VERSION 14.29.30133
# Make sure path to MSVC tools is available to Bazel
ENV BAZEL_VC "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC"

# Install bazelisk
RUN md C:\tools\bazel
COPY bazelisk.ps1 bazelisk.ps1
RUN powershell -ExecutionPolicy Bypass -File bazelisk.ps1 

RUN Remove-Item -Path "*.ps1", "vs.bat"

# Enable long path support
RUN reg.exe add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f

# Restore default shell for Windows containers.
SHELL ["cmd.exe", "/s", "/c"]
# Default to PowerShell if no other command specified.
CMD ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"] \
	\
	Write-Host 'Verifying pip install ...'; \
	python -m pip --version; \
	\
	Write-Host 'Complete.';
