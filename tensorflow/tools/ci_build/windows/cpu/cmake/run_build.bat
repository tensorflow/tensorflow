:: This script assumes the standard setup on tensorflow Jenkins windows machines.
:: It is NOT guaranteed to work on any other machine. Use at your own risk!
::
:: REQUIREMENTS:
:: * All installed in standard locations:
::   - JDK8, and JAVA_HOME set.
::   - Microsoft Visual Studio 2015 Community Edition
::   - Msys2
::   - Anaconda3
::   - CMake
:: * Before running this script, you have to set BUILD_CC_TESTS and BUILD_PYTHON_TESTS
::   variables to either "ON" or "OFF".
:: * Either have the REPO_ROOT variable set, or run this from the repository root directory.

:: Check and set REPO_ROOT
IF [%REPO_ROOT%] == [] (
  SET REPO_ROOT=%cd%
)

:: Import all bunch of variables Visual Studio needs.
CALL "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
:: Turn echo back on, above script turns it off.
ECHO ON

:: Some common variables to be shared between runs.
SET CMAKE_EXE="C:\Program Files\cmake\bin\cmake.exe"
SET SWIG_EXE="C:\swigwin-3.0.10\swig.exe"
SET PY_EXE="C:\Program Files\Anaconda3\python.exe"
SET PY_LIB="C:\Program Files\Anaconda3\libs\python35.lib"

SET CMAKE_DIR=%REPO_ROOT%\tensorflow\contrib\cmake
SET MSBUILD_EXE="C:\Program Files (x86)\MSBuild\14.0\Bin\msbuild.exe"

:: Run cmake to create Visual Studio Project files.
%CMAKE_EXE% %CMAKE_DIR% -A x64 -DSWIG_EXECUTABLE=%SWIG_EXE% -DPYTHON_EXECUTABLE=%PY_EXE% -DCMAKE_BUILD_TYPE=Release -DPYTHON_LIBRARIES=%PY_LIB% -Dtensorflow_BUILD_PYTHON_TESTS=%BUILD_PYTHON_TESTS% -Dtensorflow_BUILD_CC_TESTS=%BUILD_CC_TESTS%

:: Run msbuild in the resulting VS project files to build a pip package.
%MSBUILD_EXE% /p:Configuration=Release /maxcpucount:32 tf_python_build_pip_package.vcxproj
