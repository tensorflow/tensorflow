REM make script on Windows platform
REM usage: make [vs_project_name.vcxproj]

@echo off
REM OPTION LISTs, please modify this as your own environment
REM If these executables are in %PATH%, they can be kept empty.
set SWIG_EXECUTABLE=D:\Tools\swigwin-3.0.12\swig.exe
set PYTHON_EXE=
set GPU=OFF
set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
set MKL_HOME=d:\Tools\IntelSWTools\compilers_and_libraries\windows\
REM END OPTION LISTS
set PARENT_DIR=%~dp0
cl
if errorlevel 9009 (
  echo "Use in VS prompt"
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
  set CUDA_HOME=%CUDA_PATH_V9_0%\bin
  if not exist "%CUDA_HOME%" (
    for /F %%i in ('where cudart64_90.dll') do set CUDA_HOME=%%~dpi
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
-Dtensorflow_BUILD_PYTHON_BINDINGS=ON ^
-Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON ^
-Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
-Dtensorflow_ENABLE_SSL_SUPPORT=OFF ^
-Dtensorflow_WIN_CPU_SIMD_OPTIONS=OFF -DMKL_HOME= %MKL_HOME% ^
-Dtensorflow_ENABLE_SNAPPY_SUPPORT=ON

if errorlevel 1 (
  goto EXIT
)

:Build
if not "%1"=="" (
  if exist %~nx1 MSBuild /p:Configuration=Release /m:8 %~nx1
)

:EXIT
cd %PARENT_DIR%
