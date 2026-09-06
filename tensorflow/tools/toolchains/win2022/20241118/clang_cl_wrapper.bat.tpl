@echo off
setlocal
set REAL_CLANG_CL=C:\tools\LLVM\bin\clang-cl.exe
set MAX_TRIES_CLANG=3
set DELAY_SECONDS_CLANG=5

set CUR_TRIES_CLANG=0
:retry
set /a CUR_TRIES_CLANG+=1
%REAL_CLANG_CL% %*
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% == 0 (
  exit /b 0
)
if %CUR_TRIES_CLANG% LSS %MAX_TRIES_CLANG% (
  echo Waiting %DELAY_SECONDS_CLANG% seconds before retrying...
  timeout /t %DELAY_SECONDS_CLANG% /nobreak > nul
  goto retry
) else (
  echo Maximum number of retries reached. Exiting.
  exit /b %EXIT_CODE%
)

