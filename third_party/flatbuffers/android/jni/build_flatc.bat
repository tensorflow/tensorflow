@rem Copyright (c) 2013 Google, Inc.
@rem
@rem This software is provided 'as-is', without any express or implied
@rem warranty.  In no event will the authors be held liable for any damages
@rem arising from the use of this software.
@rem Permission is granted to anyone to use this software for any purpose,
@rem including commercial applications, and to alter it and redistribute it
@rem freely, subject to the following restrictions:
@rem 1. The origin of this software must not be misrepresented; you must not
@rem claim that you wrote the original software. If you use this software
@rem in a product, an acknowledgment in the product documentation would be
@rem appreciated but is not required.
@rem 2. Altered source versions must be plainly marked as such, and must not be
@rem misrepresented as being the original software.
@rem 3. This notice may not be removed or altered from any source distribution.
@echo off

setlocal enabledelayedexpansion

set thispath=%~dp0

rem Path to cmake passed in by caller.
set cmake=%1
rem Path to cmake project to build.
set cmake_project_path=%2

rem Newest and oldest version of Visual Studio that it's possible to select.
set visual_studio_version_max=20
set visual_studio_version_min=8

rem Determine the newest version of Visual Studio installed on this machine.
set visual_studio_version=
for /L %%a in (%visual_studio_version_max%,-1,%visual_studio_version_min%) do (
  echo Searching for Visual Studio %%a >&2
  reg query HKLM\SOFTWARE\Microsoft\VisualStudio\%%a.0 /ve 1>NUL 2>NUL
  if !ERRORLEVEL! EQU 0 (
    set visual_studio_version=%%a
    goto found_vs
  )
)
echo Unable to determine whether Visual Studio is installed. >&2
exit /B 1
:found_vs

rem Map Visual Studio version to cmake generator name.
if "%visual_studio_version%"=="8" (
  set cmake_generator=Visual Studio 8 2005
)
if "%visual_studio_version%"=="9" (
  set cmake_generator=Visual Studio 9 2008
)
if %visual_studio_version% GEQ 10 (
  set cmake_generator=Visual Studio %visual_studio_version%
)
rem Set visual studio version variable for msbuild.
set VisualStudioVersion=%visual_studio_version%.0

rem Generate Visual Studio solution.
echo Generating solution for %cmake_generator%. >&2
cd "%cmake_project_path%"
%cmake% -G"%cmake_generator%"
if %ERRORLEVEL% NEQ 0 (
  exit /B %ERRORLEVEL%
)

rem Build flatc
python %thispath%\msbuild.py flatc.vcxproj
if ERRORLEVEL 1 exit /B 1
