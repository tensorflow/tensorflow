#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

# Use a temporary directory with a short name.
export TMPDIR="C:/tmp"
mkdir -p "$TMPDIR"

# Set bash path
export BAZEL_SH=${BAZEL_SH:-"C:/tools/msys64/usr/bin/bash"}

# Set Python path for ./configure
export PYTHON_BIN_PATH="C:/Program Files/Anaconda3/python"
export PYTHON_LIB_PATH="C:/Program Files/Anaconda3/lib/site-packages"

# Set Python path for cc_configure.bzl
export BAZEL_PYTHON="C:/Program Files/Anaconda3/python"

# Set Visual Studio path
export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio 14.0"

# Add python into PATH, it's needed because gen_git_source.py uses
# '/usr/bin/env python' as a shebang
export PATH="/c/Program Files/Anaconda3:$PATH"

# Make sure we have pip in PATH
export PATH="/c/Program Files/Anaconda3/Scripts:$PATH"

# Add Cuda and Cudnn dll directories into PATH
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/bin:$PATH"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/extras/CUPTI/libx64:$PATH"
export PATH="/c/tools/cuda/bin:$PATH"

# Set the common build options on Windows
export BUILD_OPTS='--copt=-w --host_copt=-w --verbose_failures --experimental_ui'
