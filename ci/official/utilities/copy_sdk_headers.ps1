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

# Relocate MSVC and Windows SDK headers to a host-mounted directory (C:\sdk)
# to bypass wcifs.sys container filter driver sharing violations during
# highly parallel clang-cl compilation actions (b/556761966).

$headers = @(
    @{ Src = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include'; Dst = 'C:\sdk\msvc' },
    @{ Src = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include'; Dst = 'C:\sdk\vs_aux' },
    @{ Src = 'C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt'; Dst = 'C:\sdk\ucrt' },
    @{ Src = 'C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\um'; Dst = 'C:\sdk\um' },
    @{ Src = 'C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\shared'; Dst = 'C:\sdk\shared' },
    @{ Src = 'C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\winrt'; Dst = 'C:\sdk\winrt' },
    @{ Src = 'C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\cppwinrt'; Dst = 'C:\sdk\cppwinrt' },
    @{ Src = 'C:\tools\LLVM\lib\clang\18\include'; Dst = 'C:\sdk\clang' }
)

Write-Host 'Relocating SDK and MSVC headers to C:\sdk...'
foreach ($h in $headers) {
    if (Test-Path $h.Src) {
        Write-Host "Copying $($h.Src) -> $($h.Dst)..."
        & robocopy $h.Src $h.Dst /E /MT:8 /R:2 /W:1 /NP /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) {
            Write-Error "robocopy failed for $($h.Src) with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    else {
        Write-Error "Source path not found: $($h.Src)"
        exit 1
    }
}
Write-Host 'SDK headers successfully relocated to C:\sdk.'
exit 0
