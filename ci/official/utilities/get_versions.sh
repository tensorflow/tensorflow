#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# Variables containing version strings extracted from the canonical sources:
# tensorflow/core/public/version.h and tools/pip_package/setup.py.
#
# These variables aren't available by default. Scripts must source this file
# explicitly, *after* checking if update_version.py needs to be run for a
# nightly job. update_version.py affects TF_VER_SUFFIX, TF_VER_PYTHON, and
# TF_VER_FULL.

# Note: in awk, the command '/search/ {commands}' applies the commands to any line that
# matches the /search/ regular expression. "print $N" prints the Nth "field",
# where fields are strings separated by whitespace.
export TF_VER_MAJOR=$(awk '/#define TF_MAJOR_VERSION/ {print $3}' tensorflow/core/public/version.h)
export TF_VER_MINOR=$(awk '/#define TF_MINOR_VERSION/ {print $3}' tensorflow/core/public/version.h)
export TF_VER_PATCH=$(awk '/#define TF_PATCH_VERSION/ {print $3}' tensorflow/core/public/version.h)

# Note: in awk, "print $N" prints the Nth "field", where fields are strings separated
# by whitespace. The flag "-F<x>" changes the behavior such that fields are now strings
# separated by the character <x>. Therefore, -F\' (escaped single quote) means that field
# $2 in <Tensor'flow'> is <flow>. This is useful for reading string literals like below.
export TF_VER_SUFFIX=$(awk -F\" '/#define TF_VERSION_SUFFIX/ {print $2}' tensorflow/core/public/version.h)
export TF_VER_PYTHON=$(awk -F\' '/_VERSION =/ {print $2}' tensorflow/tools/pip_package/setup.py)

# Derived helper variables.
export TF_VER_SHORT="${TF_VER_MAJOR}.${TF_VER_MINOR}"
export TF_VER_FULL="${TF_VER_MAJOR}.${TF_VER_MINOR}.${TF_VER_PATCH}${TF_VER_SUFFIX}"
