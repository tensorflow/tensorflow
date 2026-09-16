@rem Copyright 2026 The TensorFlow Authors. All Rights Reserved.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem     http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem ==============================================================================

echo [WRAPPER] Starting clang-cl-wrapper.bat
echo [WRAPPER] CD=%CD%
echo [WRAPPER] BAZEL_PYTHON=%BAZEL_PYTHON%
echo [WRAPPER] PYTHON_BIN_PATH=%PYTHON_BIN_PATH%
echo [WRAPPER] PYTHON_PATH=%PYTHON_PATH%
echo [WRAPPER] ARGS=%*

set "PYTHON_EXE="

if not "%PYTHON_BIN_PATH%"=="" (
    echo [WRAPPER] Checking PYTHON_BIN_PATH: %PYTHON_BIN_PATH%
    if exist "%PYTHON_BIN_PATH%" (
        set "PYTHON_EXE=%PYTHON_BIN_PATH%"
        echo [WRAPPER] Found PYTHON_EXE via PYTHON_BIN_PATH
        goto :run
    ) else (
        echo [WRAPPER] PYTHON_BIN_PATH does not exist
    )
)

if not "%PYTHON_PATH%"=="" (
    echo [WRAPPER] Checking PYTHON_PATH: %PYTHON_PATH%
    if exist "%PYTHON_PATH%" (
        set "PYTHON_EXE=%PYTHON_PATH%"
        echo [WRAPPER] Found PYTHON_EXE via PYTHON_PATH
        goto :run
    ) else (
        echo [WRAPPER] PYTHON_PATH does not exist
    )
)

if not "%BAZEL_PYTHON%"=="" (
    echo [WRAPPER] Checking BAZEL_PYTHON: %BAZEL_PYTHON%
    if exist "%BAZEL_PYTHON%" (
        set "PYTHON_EXE=%BAZEL_PYTHON%"
        echo [WRAPPER] Found PYTHON_EXE via BAZEL_PYTHON
        goto :run
    ) else (
        echo [WRAPPER] BAZEL_PYTHON does not exist
    )
)

echo [WRAPPER] Checking C:\Python*
for /d %%i in (C:\Python*) do (
    if exist "%%~i\python.exe" (
        set "PYTHON_EXE=%%~i\python.exe"
        echo [WRAPPER] Found PYTHON_EXE: %%~i\python.exe
    )
)

if "%PYTHON_EXE%"=="" (
    echo [WRAPPER] Checking C:\Program Files\Python*
    for /d %%i in ("C:\Program Files\Python*") do (
        if exist "%%~i\python.exe" (
            set "PYTHON_EXE=%%~i\python.exe"
            echo [WRAPPER] Found PYTHON_EXE: %%~i\python.exe
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [WRAPPER] Defaulting to python.exe
    set "PYTHON_EXE=python.exe"
)

:run
echo [WRAPPER] Final PYTHON_EXE (before norm): %PYTHON_EXE%
set "PYTHON_EXE=%PYTHON_EXE:"=%"
set "PYTHON_EXE=%PYTHON_EXE:/=\%"
echo [WRAPPER] Final PYTHON_EXE (after norm): %PYTHON_EXE%
echo [WRAPPER] Running: "%PYTHON_EXE%" "%~dp0clang-cl-wrapper.py" %*
"%PYTHON_EXE%" "%~dp0clang-cl-wrapper.py" %*
