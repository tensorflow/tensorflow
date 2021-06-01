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

@REM To have reproducible builds, these dependencies should be pinned always.
@REM Prefer pinning to the same version as in setup.py
@REM First, upgrade pypi wheels
%PY_EXE% -m pip install --upgrade "setuptools<53" pip wheel
@REM NOTE: As numpy has releases that break semver guarantees and several other
@REM deps depend on numpy without an upper bound, we must install numpy before
@REM everything else.
@REM TODO(mihaimaruseac): Convert to requirements.txt
%PY_EXE% -m pip install "numpy ~= 1.19.2"
@REM Now, install the deps, as listed in setup.py
%PY_EXE% -m pip install "absl-py ~= 0.10"
%PY_EXE% -m pip install "astunparse ~= 1.6.3"
%PY_EXE% -m pip install "flatbuffers ~= 1.12.0"
%PY_EXE% -m pip install "google_pasta ~= 0.2"
%PY_EXE% -m pip install "h5py ~= 3.1.0"
%PY_EXE% -m pip install "keras_preprocessing ~= 1.1.2"
%PY_EXE% -m pip install "opt_einsum ~= 3.3.0"
%PY_EXE% -m pip install "protobuf >= 3.9.2"
%PY_EXE% -m pip install "six ~= 1.15.0"
%PY_EXE% -m pip install "termcolor ~= 1.1.0"
%PY_EXE% -m pip install "typing_extensions ~= 3.7.4"
%PY_EXE% -m pip install "wheel ~= 0.35"
%PY_EXE% -m pip install "wrapt ~= 1.12.1"
@REM We need to pin gast dependency exactly
%PY_EXE% -m pip install "gast == 0.4.0"
@REM Finally, install tensorboard and estimator
@REM Note that here we want the latest version that matches (b/156523241)
%PY_EXE% -m pip install --upgrade "tb-nightly ~= 2.4.0.a"
%PY_EXE% -m pip install --upgrade "tensorflow_estimator ~= 2.5.0"
@REM Test dependencies
%PY_EXE% -m pip install "grpcio >= 1.37.0, < 2.0"
%PY_EXE% -m pip install "portpicker ~= 1.3.1"
%PY_EXE% -m pip install "scipy ~= 1.5.2"

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
