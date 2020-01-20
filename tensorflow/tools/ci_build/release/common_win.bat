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
  SET PYTHON_DIRECTORY=Python36
)
SET PY_EXE=C:\%PYTHON_DIRECTORY%\python.exe
SET PIP_EXE=C:\%PYTHON_DIRECTORY%\Scripts\pip.exe
SET PATH=%PATH%;C:\%PYTHON_DIRECTORY%

@REM TODO(amitpatankar): Make an image with these packages and remove this.

%PIP_EXE% install setuptools --upgrade
%PIP_EXE% install future>=0.17.1 --no-deps
%PIP_EXE% install tf-estimator-nightly --no-deps
%PIP_EXE% install tb-nightly --no-deps
%PIP_EXE% install numpy --upgrade --no-deps
%PIP_EXE% install opt_einsum --upgrade
%PIP_EXE% install pandas --upgrade --no-deps
%PIP_EXE% install protobuf --upgrade --no-deps
%PIP_EXE% install keras_preprocessing==1.1.0 --upgrade --no-deps
%PIP_EXE% install wrapt --upgrade --no-deps

IF "%PYTHON_DIRECTORY%"=="Python37" (
    %PIP_EXE% install absl-py==0.5.0
    %PIP_EXE% install colorama==0.3.9
    %PIP_EXE% install cycler==0.10.0
    %PIP_EXE% install gast==0.3.2
    %PIP_EXE% install jedi==0.11.1
    %PIP_EXE% install oauth2client==4.1.2
    %PIP_EXE% install portpicker==1.2.0
    %PIP_EXE% install parso==0.1.1
    %PIP_EXE% install protobuf==3.8.0
    %PIP_EXE% install scikit-learn==0.19.2
    %PIP_EXE% install scipy==1.1.0
    %PIP_EXE% install termcolor==1.1.0
)

@REM TODO(amitpatankar): this is just a quick fix so that windows build doesn't
@REM break with gast upgrade to 0.3.2. Need to figure out the right way to
@REM handle this case.
%PIP_EXE% install gast==0.3.2
%PIP_EXE% install astunparse==1.6.3

:: Set cuda related environment variables. If we are not using CUDA, these are not used.
IF NOT DEFINED TF_CUDA_VERSION (
  SET TF_CUDA_VERSION=10.1
)
SET TF_CUDNN_VERSION=7
SET TF_CUDA_COMPUTE_CAPABILITIES=3.5,3.7,5.2,6.0,6.1,7.0
SET CUDA_TOOLKIT_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v%TF_CUDA_VERSION%
SET CUDNN_INSTALL_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v%TF_CUDA_VERSION%
SET PATH=%CUDA_TOOLKIT_PATH%\extras\CUPTI\libx64;%PATH%
SET PATH=%CUDA_TOOLKIT_PATH%\bin;%PATH%
SET PATH=%CUDNN_INSTALL_PATH%\bin;%PATH%

@REM
@REM Setup Bazel
@REM
:: Download Bazel from github and make sure its found in PATH.
SET BAZEL_VERSION=1.2.1
md C:\tools\bazel\
wget -q https://github.com/bazelbuild/bazel/releases/download/%BAZEL_VERSION%/bazel-%BAZEL_VERSION%-windows-x86_64.exe -O C:/tools/bazel/bazel.exe
SET PATH=C:\tools\bazel;%PATH%
bazel version
