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

# This Dockerfile creates an image that has:
# - Visual Studio 2022 Build Tools
# - MSVC 14.4x (Latest VS 2022 C++ Build Tools)
# - LLVM/Clang 18.1.4
# - MSYS2 + curl, git, patch, vim, unzip, zip
# - Python 3.10 - 3.15
# - Bazelisk 1.29.0
# - JDK 25 (Azul Zulu)

FROM mcr.microsoft.com/powershell:lts-servercore-ltsc2022

SHELL ["pwsh.exe", "-ExecutionPolicy", "Bypass", "-Command", \
       "$ErrorActionPreference='Stop'; $PSNativeCommandUseErrorActionPreference=$true; $ProgressPreference='SilentlyContinue'; $VerbosePreference = 'Continue';"]

RUN md C:\TEMP; md C:\TMP; md C:\setup
ENV TMP "C:/TMP"
ENV TEMP "C:/TEMP"

# Copy all scripts once to reduce image layer count
COPY scripts/ C:/setup/

# Run all installation scripts in a single layer to avoid layer bloat.
# $ErrorActionPreference='Stop' inside each script ensures
#that any failure halts the build.
RUN C:\setup\setup_system.ps1 ; \
    C:\setup\install_7zip.ps1 ; \
    C:\setup\install_vs_build_tools.ps1 ; \
    C:\setup\install_llvm.ps1 ; \
    C:\setup\install_msys2.ps1 ; \
    C:\setup\install_uv.ps1 ; \
    C:\setup\install_python.ps1 -Version '3.10.11' -TargetDir 'C:\Python310' -Symlinks @('python3.10.exe') ; \
    C:\setup\install_python.ps1 -Version '3.11.9' -TargetDir 'C:\Python311' -Symlinks @('python3.exe', 'python3.11.exe') ; \
    C:\setup\install_python.ps1 -Version '3.12.3' -TargetDir 'C:\Python312' -Symlinks @('python3.12.exe') ; \
    C:\setup\install_python.ps1 -Version '3.13.5' -TargetDir 'C:\Python313' -Symlinks @('python3.13.exe') ; \
    C:\setup\install_python.ps1 -Version '3.14.0rc1' -FtpDir '3.14.0' -TargetDir 'C:\Python314' -Symlinks @('python3.14.exe') ; \
    C:\setup\install_jdk.ps1 ; \
    C:\setup\install_bazelisk.ps1 ; \
    C:\setup\install_gcloud.ps1 ; \
    New-Item -ItemType Directory -Path C:\tools -Force | Out-Null ; \
    Copy-Item -Path C:\setup\entrypoint.ps1 -Destination C:\tools\entrypoint.ps1 ; \
    Remove-Item -Recurse -Force C:\setup -ErrorAction SilentlyContinue

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
ENV BAZEL_VC "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC"

# Environment variables to work around MSYS path translation issues.
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

ENTRYPOINT ["pwsh.exe", "-ExecutionPolicy", "Bypass", "-File", "C:\\tools\\entrypoint.ps1"]
