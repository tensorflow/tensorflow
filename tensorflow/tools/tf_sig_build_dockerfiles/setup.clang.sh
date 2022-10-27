#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# setup.clang.sh: Clone and install Clang at HEAD.

# apt-get update
# apt-get install -y cmake
git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
# TODO(juanantoniomc): Change "Debug" to "Release" after successful compilation
cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" ../llvm
make -j$(nproc)
# Move to clang to right location
mkdir -p /usr/lib/llvm-14/
mv ./bin /usr/lib/llvm-14/
mv ./lib /usr/lib/llvm-14/