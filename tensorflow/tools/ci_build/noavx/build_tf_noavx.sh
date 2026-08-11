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
echo " Starting TensorFlow CPU Build (No-AVX) "
echo " Target Architecture: Westmere (SSE4.2, AES-NI, No AVX)"
echo "=========================================================="

cd /tensorflow

# Fast CRLF normalization for packaging, CI scripts, and build definitions
if command -v dos2unix >/dev/null 2>&1; then
  echo "=== Normalizing packaging and build file line endings to LF ==="
  dos2unix -q /tensorflow/tensorflow/tools/pip_package/* 2>/dev/null || true
  dos2unix -q /tensorflow/tensorflow/tools/ci_build/noavx/* 2>/dev/null || true
  dos2unix -q /tensorflow/tensorflow/tools/toolchains/cpus/py*/BUILD* 2>/dev/null || true
  dos2unix -q /tensorflow/tensorflow/BUILD* 2>/dev/null || true
  dos2unix -q /tensorflow/configure* /tensorflow/WORKSPACE* /tensorflow/.bazelrc 2>/dev/null || true
fi

# Clean any existing configuration
rm -f .tf_configure.bazelrc

# Configure environment variables for non-interactive ./configure
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_CLANG=1
export CLANG_COMPILER_PATH=$(which clang)
export CC_OPT_FLAGS="-march=westmere -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0
export TF_ENABLE_XLA=1

python3 configure.py

echo "=== .tf_configure.bazelrc created ==="
cat .tf_configure.bazelrc

# Ensure clean embedded_tools install extraction
rm -rf /root/.cache/bazel/_bazel_root/install

echo "=== Starting Bazel Build ==="
bazel build \
  --config=opt \
  --repo_env=WHEEL_NAME=tensorflow \
  --verbose_failures \
  //tensorflow/tools/pip_package:wheel

echo "=== Copying Wheel to /tf_wheel ==="
mkdir -p /tf_wheel
cp -v bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl /tf_wheel/

echo "=== TensorFlow No-AVX Wheel Build Succeeded! ==="
ls -lh /tf_wheel/
