echo on
setlocal enableextensions enabledelayedexpansion

SET PYTHON_DIRECTORY=Python39

@REM This is the path to bazel_wrapper.py, should be set as an argument
set BAZEL_WRAPPER_PATH=%~f1

@REM Load common definitions, install bazel
CALL tensorflow\tools\ci_build\release\common_win.bat

@REM Set up common variables used through the script
set WIN_OUT=win.out
set WIN_OUT_TARGET=gen_win_out
set BUILD_PATH=tensorflow/tools/ci_build/builds
set GEN_SCRIPT=%BUILD_PATH%/%WIN_OUT_TARGET%.sh
set GEN_BUILD=%BUILD_PATH%/BUILD

@REM Run the presubmit win build.
CALL tensorflow\tools\ci_build\windows\cpu\pip\run.bat --enable_remote_cache %* > %BUILD_PATH%/%WIN_OUT% 2>&1
set RC=%errorlevel%

@REM Since we are running the sanity build remotely (rbe), we need to build a bazel
@REM target that would output the log generated above and return the expected
@REM error code.
echo package(default_visibility = ["//visibility:public"]) > %GEN_BUILD%
echo. >> %GEN_BUILD%
echo sh_test( >> %GEN_BUILD%
echo     name = "%WIN_OUT_TARGET%", >> %GEN_BUILD%
echo     srcs = ["%WIN_OUT_TARGET%.sh"], >> %GEN_BUILD%
echo     data = ["%WIN_OUT%"], >> %GEN_BUILD%
echo     tags = ["local"], >> %GEN_BUILD%
echo ) >> %GEN_BUILD%

echo #!/bin/bash > %GEN_SCRIPT%
echo function rlocation() { >> %GEN_SCRIPT%
echo   fgrep -m1 "$1 " "$RUNFILES_MANIFEST_FILE" ^| cut -d' ' -f2- >> %GEN_SCRIPT%
echo } >> %GEN_SCRIPT%
echo cat $(rlocation %BUILD_PATH%/%WIN_OUT%) >> %GEN_SCRIPT%
echo exit %RC% >> %GEN_SCRIPT%

@REM Now trigger the rbe build that outputs the log
chmod +x %GEN_SCRIPT%

@REM Run bazel test command.
%PY_EXE% %BAZEL_WRAPPER_PATH% --output_user_root=%TMPDIR% ^
  --host_jvm_args=-Dbazel.DigestFunction=SHA256 test ^
  %BUILD_PATH%:%WIN_OUT_TARGET% --test_output=all ^
  --experimental_ui_max_stdouterr_bytes=-1
