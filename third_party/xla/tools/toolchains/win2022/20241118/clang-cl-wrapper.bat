@echo off
setlocal EnableDelayedExpansion

REM This wrapper script calls clang-cl.exe with all given arguments.
REM If clang-cl.exe fails, it retries up to 5 times with 1 second delay between retries.

set CYGWIN=nodosfilewarning

set MAX_RETRIES=5
set RETRY_COUNT=0
set RETRY_DELAY_SECONDS=1
set SUCCEEDED=0
for /L %%i in (1,1,%MAX_RETRIES%) do (
  if !SUCCEEDED! == 0 (
    clang-cl.exe %*
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
