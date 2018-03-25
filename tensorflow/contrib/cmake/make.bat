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
REM The path to swig.exe, if the swig.exe is in %PATH%, leave it empty
set SWIG_EXECUTABLE=
REM The path to python.exe, if the python.exe is in %PATH%, leave it empty
set PYTHON_EXECUTABLE=
REM To build tensorflow.dll or not
set SHARED_LIB=OFF
REM To use GPU acceleration
set GPU=OFF
REM To use CPU SIMD intrinsic, options can be /arch:AVX2, /arch:AVX, /arch:SSE2 or empty
set WIN_CPU_SIMD=/arch:AVX2
REM To use MKL or not
set MKL=OFF
REM To use MKL_DNN, MKL DNN must be based on MKL
set MKL_DNN=OFF
REM Path to MKL installation directory, you must modify MKL_HOME to your own path if you turn MKL on
set MKL_HOME="c:\Program Files (x86)\IntelSWTools\compilers_and_libraries"
REM The other cmake variables are listed here
set OTHER_CMAKE_ARGS=-Dtensorflow_BUILD_PYTHON_BINDINGS=ON ^
                     -Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON ^
                     -Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
                     -Dtensorflow_ENABLE_SSL_SUPPORT=OFF ^
                     -Dtensorflow_ENABLE_SNAPPY_SUPPORT=ON

set PARENT_DIR=%~dp0
where /Q msbuild
if %errorlevel% neq 0 (
  echo msbuild is not in PATH, please use me in VS prompt
  goto EXIT
)
REM query the msbuild version, required >= VS2015
for /F "skip=2 tokens=1 delims=." %%i in ('msbuild /version') do set MSVC_VERSION=%%i
if %MSVC_VERSION%==15 set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
if %MSVC_VERSION%==14 set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
if %MSVC_VERSION% LSS 14 (
  echo "Visual Studio version too low!"
  goto EXIT
)
REM check if swig exists
if not exist "%SWIG_EXECUTABLE%" (
  for /F %%i in ('where swig') do set SWIG_EXECUTABLE=%%i
  if errorlevel 1 (
    echo "Swig is not in PATH environment"
    goto EXIT
  )
)
REM check if python exists
if not exist "%PYTHON_EXECUTABLE%" (
  for /F %%i in ('where python') do set PYTHON_EXECUTABLE=%%i
  if errorlevel 1 (
    echo "Python is not in PATH environment"
    goto EXIT
  )
)
REM find python lib
for /F "tokens=2,3 delims=. " %%i in ('%PYTHON_EXECUTABLE% -V') do set PYTHON_VER=%%i%%j
set PYTHON_LIB=%PYTHON_EXECUTABLE:~0,-10%libs\python%PYTHON_VER%.lib
if not exist %PYTHON_LIB% (
  echo "Could not find python lib as %PYTHON_LIB%"
  goto EXIT
)
REM find CUDA path
if /I "%GPU%"=="ON" (
  set CUDA_HOME=%CUDA_PATH_V9_0%
  if not exist "%CUDA_HOME%" (
    for /F %%i in ('where cudart64_90.dll') do set CUDA_HOME=%%~dpi\..\
    if errorlevel 1 (
      echo "cuda runtime is not in PATH environment"
      goto EXIT
    )
  )
  set OTHER_CMAKE_ARGS=%OTHER_CMAKE_ARGS% -DCUDNN_HOME=%CUDA_HOME%
)
REM check if MKL_HOME exists
if /I "%MKL%"=="ON" (
  if not exist %MKL_HOME% (
    echo "Your MKL_HOME does not exist, please install MKL before you turn on MKL option"
    goto EXIT
  )
  set OTHER_CMAKE_ARGS=%OTHER_CMAKE_ARGS% -DMKL_HOME=%MKL_HOME%
)

if not exist build mkdir build
cd build

if not "%1"=="" goto Build
:Generate
cmake .. -G %CMAKE_GENERATOR% -DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=%SWIG_EXECUTABLE% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DPYTHON_LIBRARIES=%PYTHON_LIB% ^
-Dtensorflow_ENABLE_GPU=%GPU% ^
-Dtensorflow_BUILD_SHARED_LIB=%SHARED_LIB% ^
-Dtensorflow_WIN_CPU_SIMD_OPTIONS=%WIN_CPU_SIMD% ^
-Dtensorflow_ENABLE_MKL_SUPPORT=%MKL% ^
-Dtensorflow_ENABLE_MKLDNN_SUPPORT=%MKL_DNN% ^
%OTHER_CMAKE_ARGS%

if errorlevel 1 (
  goto EXIT
)

:Build
REM /m:8 is using 8 threads to compile, you can change this number to fit for your environment
if not "%1"=="" (
  if exist %~nx1 MSBuild /p:Configuration=Release /m:8 %~nx1
)

:EXIT
cd %PARENT_DIR%
