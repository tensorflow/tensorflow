:: Copyright 2026 The TensorFlow Authors. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
:: ==============================================================================
@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo  Starting TensorFlow Windows CPU Build (No-AVX)
echo  Target Architecture: x86-64 / SSE2 (No AVX, No RBE)
echo ==========================================================

del /f /q .tf_configure.bazelrc 2>nul

if exist "C:\tools\LLVM\bin\clang.exe" (
  set CLANG_COMPILER_PATH=C:/tools/LLVM/bin/clang.exe
) else if exist "C:\Program Files\LLVM\bin\clang.exe" (
  set CLANG_COMPILER_PATH=C:/Program Files/LLVM/bin/clang.exe
) else (
  set CLANG_COMPILER_PATH=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe
)

set TF_NEED_ROCM=0
set TF_NEED_CUDA=0
set TF_NEED_CLANG=1
set CC_OPT_FLAGS=-march=westmere -Wno-sign-compare -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set TF_SET_ANDROID_WORKSPACE=0
set TF_ENABLE_XLA=1

if exist "C:\tools\msys64\usr\bin\bash.exe" (
  set BAZEL_SH=C:\tools\msys64\usr\bin\bash.exe
)

python configure.py
if errorlevel 1 exit /b 1

echo === Cleaning Up Stale Temp Files and Unused Toolchains ===
powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'C:\hostedtoolcache', 'C:\Program Files\dotnet', 'C:\Program Files\Android', 'C:\Program Files (x86)\Android', 'C:\ProgramData\Package Cache'" 2>nul
del /f /s /q C:\TMP\* 2>nul
del /f /s /q C:\Temp\* 2>nul

echo === Starting Bazel Build for Windows Wheel (No-AVX) ===
bazel --output_user_root=C:\x build --config=opt --config=nogcp --config=win_clang --copt=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --host_copt=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --copt=-g0 --host_copt=-g0 --strip=always --linkopt=clang_rt.builtins-x86_64.lib --host_linkopt=clang_rt.builtins-x86_64.lib --linkopt=/DEBUG:NONE --host_linkopt=/DEBUG:NONE --nobuild_runfile_links --repo_env=WHEEL_NAME=tensorflow --verbose_failures //tensorflow/tools/pip_package:wheel
if errorlevel 1 exit /b 1

echo === Windows No-AVX Wheel Build Succeeded! ===
if not exist "build_output" mkdir build_output
for /r bazel-bin\tensorflow\tools\pip_package %%f in (*.whl) do (
  copy "%%f" build_output\
)
dir build_output
