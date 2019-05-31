%echo off

cd /d %~dp0

if exist _build rd /s /q _build

mkdir _build
chdir _build


rem cmake ../ -G "Visual Studio 15 Win64" -DCMAKE_GENERATOR_TOOLSET=v141,host=x64 -DCMAKE_INSTALL_PREFIX:PATH=.\install

CALL :NORMALIZEPATH "..\..\..\.."
SET SOURCE_DIR=%RETVAL%

echo %SOURCE_DIR%

SET SOURCE_DIR=F:\frameworks\tensorflow\

CALL :NORMALIZEPATH "../../../tools/git/gen_git_source.py"
SET SOURCE_PYTHON_SCRIPT=%RETVAL%

CALL :NORMALIZEPATH "../../../core/util/version_info.cc"
SET SOURCE_VERSION_CC=%RETVAL%

python %SOURCE_PYTHON_SCRIPT% --raw_generate %SOURCE_VERSION_CC% --source_dir %SOURCE_DIR% --git_tag_override=

cmake ../ -G "Visual Studio 15 Win64" -DCMAKE_GENERATOR_TOOLSET=v141,host=x64 -DCMAKE_INSTALL_PREFIX:PATH=.\install

EXIT /B

:NORMALIZEPATH
  SET RETVAL=%~dpfn1
  EXIT /B



                                                                              