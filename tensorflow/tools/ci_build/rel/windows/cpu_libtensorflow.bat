:: Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

SET TF_DIR=%cd%\github\tensorflow
SET TF_DOCKER_DIR=C:\src\tensorflow
REM TODO(belitskiy): Switch to Artifact Registry
set TF_DOCKER_IMAGE="gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:1082ef4299a72e44a84388f192ecefc81ec9091c146f507bc36070c089c0edcc"

docker pull %TF_DOCKER_IMAGE% || exit /b 1
@echo *****Finished docker image pull: %date% %time%

docker run ^
    --name tf ^
    -itd ^
    --env TF_PYTHON_VERSION=%TF_PYTHON_VERSION% ^
    -v %TF_DIR%:%TF_DOCKER_DIR% ^
    -v T:\tmp:C:\tmp ^
    -w %TF_DOCKER_DIR% ^
    -e GOOGLE_APPLICATION_CREDENTIALS=%GUESTKEYNAME% ^
    --dns 8.8.8.8 ^
    --dns 8.8.4.4 ^
    %TF_DOCKER_IMAGE% ^
    bash || exit /b 1

docker exec tf ^
    bash tensorflow/tools/ci_build/windows/libtensorflow_cpu.sh || exit /b 1

gsutil cp %TF_DIR%\windows_cpu_libtensorflow_binaries.tar.gz ^
  gs://libtensorflow-nightly/prod/tensorflow/release/windows/latest/cpu
