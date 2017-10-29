#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

LLVM_SVN_REVISION="314281"
CLANG_TMP_DIR=/tmp/clang-build

mkdir "$CLANG_TMP_DIR"

pushd "$CLANG_TMP_DIR"

# Checkout llvm+clang
svn co -q -r$LLVM_SVN_REVISION http://llvm.org/svn/llvm-project/llvm/trunk "$CLANG_TMP_DIR/llvm"
svn co -q -r$LLVM_SVN_REVISION http://llvm.org/svn/llvm-project/cfe/trunk "$CLANG_TMP_DIR/llvm/tools/clang"

# Build 1st stage. Compile clang with system compiler
mkdir "$CLANG_TMP_DIR/build-1"
cd "$CLANG_TMP_DIR/build-1"
cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release "$CLANG_TMP_DIR/llvm"
make -j `nproc` clang clang-headers

# Build 2nd stage. Compile clang with clang built in stage 1
mkdir "$CLANG_TMP_DIR/build-2"
cd "$CLANG_TMP_DIR/build-2"

CC="$CLANG_TMP_DIR/build-1/bin/clang" \
CXX="$CLANG_TMP_DIR/build-1/bin/clang++" \
cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local "$CLANG_TMP_DIR/llvm"

make -j `nproc` install-clang install-clang-headers

popd

# Cleanup
rm -rf "$CLANG_TMP_DIR"
