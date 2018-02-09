@echo off
set PARENT_DIR=%~dp0
cl
if errorlevel 9009 (
  echo "Use in VS prompt"
  goto EXIT
)
if not exist build mkdir build
cd build

if not "%1"=="" goto Build
:Generate
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=D:\Tools\swigwin-3.0.12\swig.exe ^
-DPYTHON_EXECUTABLE=C:\Users\%username%\AppData\Local\Programs\Python\Python36\python.exe ^
-DPYTHON_LIBRARIES=C:\Users\%username%\AppData\Local\Programs\Python\Python36\libs\python36.lib ^
-Dtensorflow_ENABLE_GPU=OFF ^
-DCUDNN_HOME=D:\Works\cuda\bin\v9.0 ^
-Dtensorflow_BUILD_PYTHON_BINDINGS=ON ^
-Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON ^
-Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
-Dtensorflow_ENABLE_SSL_SUPPORT=OFF ^
-Dtensorflow_WIN_CPU_SIMD_OPTIONS=ON ^
-DCPU_SIMD=AVX2 ^
-DMKL_HOME=d:\softwares\Tools\IntelSWTools\compilers_and_libraries\windows\ ^
-Dtensorflow_ENABLE_SNAPPY_SUPPORT=ON ^
-Dtensorflow_DECOUPLE_EXTERNAL_DEPENDENCY=ON

if errorlevel 1 (
  goto EXIT
)

:Build
if not "%1"=="" (
  if exist %~nx1 MSBuild /p:Configuration=Release /m:8 %~nx1
)

:EXIT
cd %PARENT_DIR%
