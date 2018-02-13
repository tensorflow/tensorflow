@echo off
REM # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
REM #
REM # Licensed under the Apache License, Version 2.0 (the "License");
REM # you may not use this file except in compliance with the License.
REM # You may obtain a copy of the License at
REM #
REM #     http://www.apache.org/licenses/LICENSE-2.0
REM #
REM # Unless required by applicable law or agreed to in writing, software
REM # distributed under the License is distributed on an "AS IS" BASIS,
REM # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM # See the License for the specific language governing permissions and
REM # limitations under the License.
REM # ==============================================================================

REM make script on Windows platform
REM usage: make [vs_project_name.vcxproj]

REM OPTION LISTs, please modify this as your own environment
REM If these executables are in %PATH%, they can be kept empty.
REM Possible WIN_CPU_SIMD options: /arch:[AVX2, AVX, SSE2, SSE]
set SWIG_EXECUTABLE=D:\Tools\swigwin-3.0.12\swig.exe
set PYTHON_EXE=
set SHARED_LIB=ON
set GPU=OFF
set WIN_CPU_SIMD=/arch:AVX2
set MKL=ON
set MKL_DNN=ON
set MKL_HOME=d:\softwares\Tools\IntelSWTools\compilers_and_libraries
set OTHER_CMAKE_ARGS=-Dtensorflow_BUILD_PYTHON_BINDINGS=ON ^
                     -Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON ^
                     -Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
                     -Dtensorflow_ENABLE_SSL_SUPPORT=OFF ^
                     -Dtensorflow_ENABLE_SNAPPY_SUPPORT=ON
REM END OPTION LISTS
set PARENT_DIR=%~dp0
for /F "skip=2 tokens=1 delims=." %%i in ('msbuild /version') do set MSVC_VERSION=%%i
if %errorlevel% neq 0 (
  echo "msbuild is not in PATH, please use me in VS prompt"
  goto EXIT
) else if %MSVC_VERSION%==15 (
  set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
) else if %MSVC_VERSION%==14 (
  set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
) else (
  echo "Visual Studio version too low!"
  goto EXIT
)
if not exist "%SWIG_EXECUTABLE%" (
  for /F %%i in ('where swig') do set SWIG_EXECUTABLE=%%i
  if errorlevel 1 (
    echo "Swig is not in PATH environment"
    goto EXIT
  )
)
if not exist "%PYTHON_EXE%" (
  for /F %%i in ('where python') do set PYTHON_EXE=%%i
  if errorlevel 1 (
    echo "Python is not in PATH environment"
    goto EXIT
  )
)
for /F "tokens=2,3 delims=. " %%i in ('%PYTHON_EXE% -V') do set PYTHON_VER=%%i%%j
set PYTHON_LIB=%PYTHON_EXE:~0,-10%libs\python%PYTHON_VER%.lib
if not exist %PYTHON_LIB% (
  echo "Could not find python lib as %PYTHON_LIB%"
  goto EXIT
)
if /I "%GPU%"=="ON" (
  set CUDA_HOME=%CUDA_PATH_V9_0%
  if not exist "%CUDA_HOME%" (
    for /F %%i in ('where cudart64_90.dll') do set CUDA_HOME=%%~dpi\..\
    if errorlevel 1 (
      echo "cuda runtime is not in PATH environment"
      goto EXIT
    )
  )
)

if not exist build mkdir build
cd build

if not "%1"=="" goto Build
:Generate
cmake .. -G %CMAKE_GENERATOR% -DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=%SWIG_EXECUTABLE% -DPYTHON_EXECUTABLE=%PYTHON_EXE% -DPYTHON_LIBRARIES=%PYTHON_LIB% ^
-Dtensorflow_ENABLE_GPU=%GPU% -DCUDNN_HOME=%CUDA_HOME% ^
-Dtensorflow_BUILD_SHARED_LIB=%SHARED_LIB% ^
-Dtensorflow_WIN_CPU_SIMD_OPTIONS=%WIN_CPU_SIMD% ^
-Dtensorflow_ENABLE_MKL_SUPPORT=%MKL% ^
-Dtensorflow_ENABLE_MKLDNN_SUPPORT=%MKL_DNN% ^
-DMKL_HOME=%MKL_HOME% ^
%OTHER_CMAKE_ARGS%

if errorlevel 1 (
  goto EXIT
)

:Build
if not "%1"=="" (
  if exist %~nx1 MSBuild /p:Configuration=Release /m:8 %~nx1
)

:EXIT
cd %PARENT_DIR%
