#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Usage: basic_mkl_test.sh

# Helper function to traverse directories up until given file is found.
function upsearch () {
  test / == "$PWD" && return || \
      test -e "$1" && echo "$PWD" && return || \
      cd .. && upsearch "$1"
}

# Set up WORKSPACE.
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"

BUILD_TAG=mkl-gpu-ci-test CI_BUILD_USER_FORCE_BADNAME=yes ${WORKSPACE}/tensorflow/tools/ci_build/ci_build.sh gpu tensorflow/tools/ci_build/linux/gpu/run_mkl.sh
