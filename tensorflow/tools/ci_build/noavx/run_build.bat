@echo off
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

setlocal

echo ==========================================================
echo  TensorFlow CPU No-AVX Hermetic Docker Builder (Windows)
echo ==========================================================

:: Resolve script and repository root directories safely
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Resolve repository root (4 levels up from tools\ci_build\noavx)
for %%I in ("%SCRIPT_DIR%\..\..\..\..") do set "TF_ROOT=%%~fI"

if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=%TF_ROOT%\build_output"
if "%CACHE_DIR%"=="" set "CACHE_DIR=%TF_ROOT%\.bazel_cache"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"

echo Script Directory: %SCRIPT_DIR%
echo Repository Root:  %TF_ROOT%
echo Output Directory: %OUTPUT_DIR%

echo [1/3] Building hermetic Docker image 'tf-cpu-noavx-builder'...
docker build -t tf-cpu-noavx-builder -f "%SCRIPT_DIR%\Dockerfile" "%SCRIPT_DIR%"
if errorlevel 1 (
    echo [ERROR] Docker build failed. Make sure Docker Desktop is running.
    exit /b 1
)

echo [2/3] Building TensorFlow CPU No-AVX Wheel inside Docker container...
docker run --rm --name tf-cpu-noavx-build-run -v "%TF_ROOT%:/tensorflow" -v "%OUTPUT_DIR%:/tf_wheel" -v "%CACHE_DIR%:/root/.cache" tf-cpu-noavx-builder
if errorlevel 1 (
    echo [ERROR] TensorFlow build inside Docker container failed.
    exit /b 1
)

echo [3/3] Verifying built wheel inside Docker container...
docker run --rm -v "%OUTPUT_DIR%:/tf_wheel:ro" tf-cpu-noavx-builder bash -c "pip install --force-reinstall /tf_wheel/*.whl && python3 -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__); a = tf.constant([[1.0, 2.0], [3.0, 4.0]]); b = tf.constant([[1.0, 1.0], [0.0, 1.0]]); c = tf.matmul(a, b); print('MatMul result:\n', c); assert float(tf.reduce_sum(c).numpy()) == 14.0; print('SUCCESS: Wheel passed runtime verification!')\""
if errorlevel 1 (
    echo [ERROR] Wheel verification failed.
    exit /b 1
)

echo ==========================================================
echo  SUCCESS: TensorFlow No-AVX Wheel built and verified!
echo  Output wheel location: %OUTPUT_DIR%
echo ==========================================================
dir "%OUTPUT_DIR%\*.whl"

endlocal
