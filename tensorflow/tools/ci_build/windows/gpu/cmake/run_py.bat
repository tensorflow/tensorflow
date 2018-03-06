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

:: Record the directory we are in. Script should be invoked from the root of the repository.
SET REPO_ROOT=%cd%

:: Make sure we have a clean directory to build things in.
SET BUILD_DIR=cmake_build
RMDIR %BUILD_DIR% /S /Q
MKDIR %BUILD_DIR%
CD %BUILD_DIR%

:: Set which tests to build
IF DEFINED BUILD_CC_TESTS (ECHO BUILD_CC_TESTS is set to %BUILD_CC_TESTS%) ELSE (SET BUILD_CC_TESTS=OFF)
IF DEFINED BUILD_PYTHON_TESTS (ECHO BUILD_PYTHON_TESTS is set to %BUILD_PYTHON_TESTS%) ELSE (SET BUILD_PYTHON_TESTS=ON)

:: Set if this build is a nightly
IF DEFINED TF_NIGHTLY (ECHO TF_NIGHTLY is set to %TF_NIGHTLY%) ELSE (SET TF_NIGHTLY=OFF)

:: Set pip binary location. Do not override if it is set already.
IF DEFINED PIP_EXE (ECHO PIP_EXE is set to %PIP_EXE%) ELSE (SET PIP_EXE="C:\Program Files\Anaconda3\Scripts\pip.exe")

:: Run the CMAKE build to build the pip package.
CALL %REPO_ROOT%\tensorflow\tools\ci_build\windows\gpu\cmake\run_build.bat
if %errorlevel% neq 0 exit /b %errorlevel%

:: Since there are no wildcards in windows command prompt, use dark magic to get the wheel file name.
DIR %REPO_ROOT%\%BUILD_DIR%\tf_python\dist\ /S /B > wheel_filename_file
set /p WHEEL_FILENAME=<wheel_filename_file
del wheel_filename_file

:: Install absl-py.
%PIP_EXE% install --upgrade absl-py

:: Install the pip package.
echo Installing PIP package...
%PIP_EXE% install --upgrade --no-deps %WHEEL_FILENAME% -v -v
if %errorlevel% neq 0 exit /b %errorlevel%

:: Run all python tests if the installation succeeded.
echo Running tests...
ctest -C Release --output-on-failure --jobs 1
