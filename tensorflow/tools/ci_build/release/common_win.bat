:: Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
:: =============================================================================
echo on

@REM
@REM Set Environment Variables
@REM
IF NOT DEFINED PYTHON_DIRECTORY (
  SET PYTHON_DIRECTORY=Python37
)
SET PY_EXE=C:\%PYTHON_DIRECTORY%\python.exe
SET PATH=%PATH%;C:\%PYTHON_DIRECTORY%

@REM First, upgrade pypi wheels
%PY_EXE% -m pip install --upgrade "setuptools<53" pip wheel

@REM NOTE: Windows doesn't have any additional requirements from the common ones.
%PY_EXE% -m pip install -r tensorflow/tools/ci_build/release/requirements_common.txt

:: Set cuda related environment variables. If we are not using CUDA, these are not used.
IF NOT DEFINED TF_CUDA_VERSION (
  SET TF_CUDA_VERSION=11.2
)
IF NOT DEFINED TF_CUDNN_VERSION (
  SET TF_CUDNN_VERSION=8
)
SET TF_CUDA_COMPUTE_CAPABILITIES=sm_35,sm_50,sm_60,sm_70,sm_75,compute_80
SET CUDA_TOOLKIT_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v%TF_CUDA_VERSION%
SET CUDNN_INSTALL_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v%TF_CUDA_VERSION%
SET PATH=%CUDA_TOOLKIT_PATH%\extras\CUPTI\libx64;%PATH%
SET PATH=%CUDA_TOOLKIT_PATH%\bin;%PATH%
SET PATH=%CUDNN_INSTALL_PATH%\bin;%PATH%

@REM
@REM Setup Bazel
@REM
:: Download Bazel from github and make sure its found in PATH.
SET BAZEL_VERSION=3.7.2
md C:\tools\bazel\
wget -q https://github.com/bazelbuild/bazel/releases/download/%BAZEL_VERSION%/bazel-%BAZEL_VERSION%-windows-x86_64.exe -O C:/tools/bazel/bazel.exe
SET PATH=C:\tools\bazel;%PATH%
bazel version
