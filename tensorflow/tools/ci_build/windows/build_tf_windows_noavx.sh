#!/usr/bin/env bash
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
set -ex

echo "=========================================================="
echo " Starting TensorFlow Windows CPU Build (No-AVX) "
echo " Target Architecture: x86-64 / SSE2 (No AVX, No RBE)"
echo "=========================================================="

# Clean any existing configuration
rm -f .tf_configure.bazelrc

# Determine Python binary and paths
PYTHON_BIN=$(which python.exe 2>/dev/null || which python 2>/dev/null || echo "python")
export PYTHON_BIN_PATH=$($PYTHON_BIN -c "import sys; print(sys.executable)" 2>/dev/null || echo "python")
if command -v cygpath >/dev/null 2>&1; then
  export PYTHON_BIN_PATH=$(cygpath -m "$PYTHON_BIN_PATH")
fi
export PYTHON_LIB_PATH=$($PYTHON_BIN -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if command -v cygpath >/dev/null 2>&1 && [ -n "$PYTHON_LIB_PATH" ]; then
  export PYTHON_LIB_PATH=$(cygpath -m "$PYTHON_LIB_PATH")
fi

# Locate Clang on Windows runner and ensure Windows path format
if command -v clang.exe >/dev/null 2>&1; then
  CLANG_PATH=$(which clang.exe)
  if command -v cygpath >/dev/null 2>&1; then
    CLANG_PATH=$(cygpath -m "$CLANG_PATH")
  fi
  export CLANG_COMPILER_PATH="$CLANG_PATH"
elif [ -f "C:/tools/LLVM/bin/clang.exe" ]; then
  export CLANG_COMPILER_PATH="C:/tools/LLVM/bin/clang.exe"
elif [ -f "C:/Program Files/LLVM/bin/clang.exe" ]; then
  export CLANG_COMPILER_PATH="C:/Program Files/LLVM/bin/clang.exe"
elif [ -f "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe" ]; then
  export CLANG_COMPILER_PATH="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe"
fi

export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_CLANG=1
export CC_OPT_FLAGS="-march=westmere -Wno-sign-compare -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
export TF_SET_ANDROID_WORKSPACE=0
export TF_ENABLE_XLA=1

# Configure bash path for Bazel on Windows if MSYS2 exists
if [ -f "C:/tools/msys64/usr/bin/bash.exe" ]; then
  export BAZEL_SH="C:/tools/msys64/usr/bin/bash.exe"
fi

$PYTHON_BIN configure.py

echo "=== .tf_configure.bazelrc created ==="
cat .tf_configure.bazelrc

echo "=== Disk Space Before Cleanup ==="
df -h || true
powershell -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name='Free(GB)';Expression={[math]::round(\$_.Free/1GB,2)}}, @{Name='Used(GB)';Expression={[math]::round(\$_.Used/1GB,2)}}" || true

echo "=== Cleaning Up Stale Files, Runner Caches, and Unused Toolchains ==="
powershell -Command "
  @('C:\hostedtoolcache',
    'C:\tools\google-cloud-sdk',
    'C:\openjdk',
    'C:\tools\uv',
    'C:\tools\php',
    'C:\tools\ruby*',
    'C:\tools\rust*',
    'C:\Rust',
    'C:\Strawberry',
    'C:\Python3.10',
    'C:\Python3.11',
    'C:\Python3.12',
    'C:\Python3.13',
    'C:\Python3.15',
    'C:\Program Files\dotnet',
    'C:\Program Files (x86)\dotnet',
    'C:\Program Files\Android',
    'C:\Program Files (x86)\Android',
    'C:\Program Files\Java',
    'C:\Program Files\Eclipse Adoptium',
    'C:\Program Files\Zulu',
    'C:\Program Files\PostgreSQL',
    'C:\Program Files\MySQL',
    'C:\Program Files\MongoDB',
    'C:\Program Files\Google\Chrome',
    'C:\Program Files\Mozilla Firefox',
    'C:\Program Files (x86)\Microsoft\Edge',
    'C:\Program Files (x86)\Microsoft\EdgeUpdate',
    'C:\Program Files\Julia',
    'C:\vcpkg',
    'C:\Program Files\nodejs',
    'C:\Program Files\Microsoft SQL Server',
    'C:\Program Files (x86)\Microsoft SQL Server',
    'C:\Program Files\Amazon',
    'C:\Program Files\Azure*',
    'C:\Program Files\Docker',
    'C:\ProgramData\Docker',
    'C:\ProgramData\Chocolatey',
    'C:\ProgramData\Package Cache',
    'C:\Users\runneradmin\.cargo',
    'C:\Users\runneradmin\.rustup',
    'C:\Users\runneradmin\.nuget',
    'C:\Users\runneradmin\.dotnet',
    'C:\Users\runneradmin\.gradle',
    'C:\Users\runneradmin\.m2') | ForEach-Object {
      if (Test-Path \$_) {
        Write-Output \"Removing \$_\"
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue \$_
      }
    }
" || true
rm -rf /c/TMP/* /c/Temp/* /tmp/* /c/Users/*/AppData/Local/Temp/* /c/actions-runner/_work/_temp/* /c/ProgramData/Package\ Cache/* /c/Windows/Temp/* /c/tools/msys64/var/cache/* 2>/dev/null || true
rm -rf /c/Users/*/.cargo /c/Users/*/.rustup /c/Users/*/.nuget /c/Users/*/.dotnet /c/Users/*/.gradle /c/Users/*/.m2 2>/dev/null || true
rm -rf .git/objects .git/logs .git/hooks 2>/dev/null || true

echo "=== Disk Space After Cleanup ==="
df -h || true
powershell -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name='Free(GB)';Expression={[math]::round(\$_.Free/1GB,2)}}, @{Name='Used(GB)';Expression={[math]::round(\$_.Used/1GB,2)}}" || true

echo "=== Starting Bazel Build for Windows Wheel (No-AVX, Local Build) ==="
bazel --output_user_root="C:/x" build \
  --config=opt \
  --config=nogcp \
  --config=win_clang \
  --copt=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH \
  --host_copt=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH \
  --copt=-g0 \
  --host_copt=-g0 \
  --copt=/Gy \
  --host_copt=/Gy \
  --copt=/Gw \
  --host_copt=/Gw \
  --copt=/DNDEBUG \
  --host_copt=/DNDEBUG \
  --strip=always \
  --features=-generate_pdb_file \
  --features=-separate_debug_info \
  --discard_analysis_cache \
  --repository_cache="" \
  --linkopt=clang_rt.builtins-x86_64.lib \
  --host_linkopt=clang_rt.builtins-x86_64.lib \
  --linkopt=/DEBUG:NONE \
  --host_linkopt=/DEBUG:NONE \
  --linkopt=/INCREMENTAL:NO \
  --host_linkopt=/INCREMENTAL:NO \
  --linkopt=/OPT:REF \
  --host_linkopt=/OPT:REF \
  --linkopt=/OPT:ICF \
  --host_linkopt=/OPT:ICF \
  --nobuild_runfile_links \
  --repo_env=WHEEL_NAME=tensorflow \
  --verbose_failures \
  //tensorflow/tools/pip_package:wheel

echo "=== Windows No-AVX Wheel Build Complete ==="
mkdir -p /tmp/tf_wheel ./build_output
find ./bazel-bin/tensorflow/tools/pip_package -name "*.whl" -exec cp -v {} /tmp/tf_wheel/ \; 2>/dev/null || true
find ./bazel-bin/tensorflow/tools/pip_package -name "*.whl" -exec cp -v {} ./build_output/ \; 2>/dev/null || true
ls -lh ./build_output/
