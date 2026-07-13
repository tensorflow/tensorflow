# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

. "$PSScriptRoot\common.ps1"

Write-Output 'Configuring system settings...'

# Enable long paths in Windows registry
Write-Output 'Enabling Windows Long Paths...'
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
    -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force | Out-Null

# Symlink a directory, to have it pretend be the T:\ drive.
# This drive letter is used by internal CI,
# and part of it is mounted to the container during container's creation.
#
# While the mount argument (`-v host_path:container_path`) still requires
# `container_path` to be a legitimate C:\ path, in this case, 'C:\drive_t',
# this symlink does allow for the convenience of passing unedited paths
# to `docker exec` commands, e.g., 'T:\path', instead of 'C:\path',
# without having to replace the drive letter with C:\ every time.
# Such a workaround is not required on Linux, since it
# can create arbitrary paths within the container, e.g., '/t'.
# Note: This does not affect/work for `docker cp` commands.
Write-Output 'Creating C:\drive_t directory and virtual drive mapping T: ...'
New-Item -ItemType Directory -Path 'C:\drive_t' -Force | Out-Null
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\DOS Devices' -Name 'T:' -Value '\??\C:\drive_t' -PropertyType String -Force | Out-Null

Write-Output 'System configuration complete.'
