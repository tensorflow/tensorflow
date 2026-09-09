@echo off
setlocal EnableDelayedExpansion

REM This wrapper script calls clang-cl.exe with all given arguments.
REM If clang-cl.exe fails, it retries up to 5 times with 1 second delay between retries.
set CLANG_CL_PATH=
for %%i in (clang-cl.exe) do set CLANG_CL_PATH=%%~$PATH:i
if "%CLANG_CL_PATH%"=="" (
  echo Cannot find clang-cl.exe in PATH %PATH%
  exit 1
)
set LLVM_INSTALL_PATH=%CLANG_CL_PATH%
set LLVM_INSTALL_PATH=!LLVM_INSTALL_PATH:\bin\clang-cl.exe=!

set PATH=!LLVM_INSTALL_PATH!\bin;%PATH%

set CYGWIN=nodosfilewarning
set ARGS=
set SKIP_NEXT=0

for %%x in (%*) do (
  if !SKIP_NEXT! == 1 (
    set SKIP_NEXT=0
  ) else (
    set ARG=%%x
    set ARG=!ARG:"=!
    if "!ARG:~0,5!" == "/imsvc" (
      set SKIP_NEXT=1
    ) else if "!ARG:~0,1!" == "/" (
      if exist "!ARG:~1!" (
        call :TO_WINDOWS_PATH "!ARG:~1!" ARG
        set ARG=/!ARG!
      )
      call set ARGS=!ARGS! "!ARG!"
    ) else if exist "!ARG!" (
      call :TO_WINDOWS_PATH "!ARG!" ARG
      call set ARGS=!ARGS! "!ARG!"
    ) else (
      call set ARGS=!ARGS! "%%x"
    )
  )
)

set MAX_RETRIES=5
set RETRY_COUNT=0
set RETRY_DELAY_SECONDS=1
set SUCCEEDED=0
for /L %%i in (1,1,%MAX_RETRIES%) do (
  if !SUCCEEDED! == 0 (
    clang-cl.exe !ARGS!
    if !ERRORLEVEL! equ 0 (
      set SUCCEEDED=1
    ) else (
      echo Attempt %%i failed with errorlevel !ERRORLEVEL!. Retrying in !RETRY_DELAY_SECONDS! seconds...
      timeout /t !RETRY_DELAY_SECONDS! >nul
    )
  )
)

if !SUCCEEDED! == 1 (
  exit 0
) else (
  echo All clang-cl.exe attempts failed.
  exit 1
)

:TO_WINDOWS_PATH
for %%i in (%1) do set %2=%%~fi
exit /b
