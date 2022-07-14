#!/bin/bash -eu
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Clang is not available for Ubuntu 20.04 in Google mirror so we download the
# official release. Manylinux2014 RBE Docker container uses Ubuntu 20.04 as a
# base image because it installs gcc 9.3.1 that requires a newer version of the
# GNU assembler.
CLANG_MAJOR_VERSION=$(echo ${CLANG_VERSION}  | grep -o "[^.]*" | head -1)
INSTALL_DIR="/clang${CLANG_MAJOR_VERSION}"
CLANG_ARCHIVE="https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANG_VERSION}/clang+llvm-${CLANG_VERSION}-x86_64-linux-gnu-ubuntu-20.04.tar.xz"
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"
wget "${CLANG_ARCHIVE}"
tar -xJvf clang+llvm-${CLANG_VERSION}-x86_64-linux-gnu-ubuntu-20.04.tar.xz --strip=1
rm clang+llvm-${CLANG_VERSION}-x86_64-linux-gnu-ubuntu-20.04.tar.xz
