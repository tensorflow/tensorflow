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
  SET PYTHON_DIRECTORY=Python39
)
SET PY_EXE=C:\%PYTHON_DIRECTORY%\python.exe
SET PATH=%PATH%;C:\%PYTHON_DIRECTORY%

@REM First, upgrade pypi wheels
%PY_EXE% -m pip install --upgrade "setuptools" pip wheel

@REM NOTE: Windows doesn't have any additional requirements from the common ones.
%PY_EXE% -m pip install -r tensorflow/tools/ci_build/release/requirements_common.txt

@REM
@REM Setup Bazelisk
@REM
:: Download Bazelisk from GitHub and make sure its found in PATH.
md C:\tools\bazel\
wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-windows-amd64.exe -O C:/tools/bazel/bazel.exe
SET PATH=C:\tools\bazel;%PATH%
bazel version