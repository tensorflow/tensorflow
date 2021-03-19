#!/usr/bin/env bash
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

# This script can be used to initiate a bazel build with a reduced set of
# downloads, but still sufficient to test all the TFLM targets.
#
# This is primarily intended for use from a Docker image as part of the TFLM
# github continuous integration system. There are still a number of downloads
# (e.g. java) that are not necessary and it may be possible to further reduce
# the set of external libraries and downloads.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# Replace Tensorflow's versions of files with pared down versions that download
# a smaller set of external dependencies to speed up the CI.
readable_run rm -f .bazelrc
readable_run rm -f tensorflow/BUILD
readable_run rm -f tensorflow/tensorflow.bzl
readable_run rm -f tensorflow/workspace.bzl
readable_run rm -f tensorflow/workspace0.bzl
readable_run rm -f tensorflow/workspace1.bzl
readable_run rm -f tensorflow/workspace2.bzl
readable_run rm -f WORKSPACE

readable_run cp tensorflow/lite/micro/tools/ci_build/tflm_bazel/dot_bazelrc .bazelrc
readable_run cp tensorflow/lite/micro/tools/ci_build/tflm_bazel/BUILD tensorflow/
readable_run cp tensorflow/lite/micro/tools/ci_build/tflm_bazel/tensorflow.bzl tensorflow/
readable_run cp tensorflow/lite/micro/tools/ci_build/tflm_bazel/workspace.bzl tensorflow/
readable_run cp tensorflow/lite/micro/tools/ci_build/tflm_bazel/WORKSPACE ./

# Now that we are set up to download fewer external deps as part of a bazel
# build, we can go ahead and invoke bazel.

CC=clang readable_run bazel test tensorflow/lite/micro/... \
  --test_tag_filters=-no_oss --build_tag_filters=-no_oss \
  --test_output=errors

CC=clang readable_run bazel test tensorflow/lite/micro/... \
  --config=msan \
  --test_tag_filters=-no_oss,-nomsan --build_tag_filters=-no_oss,-nomsan \
  --test_output=errors

CC=clang readable_run bazel test tensorflow/lite/micro/... \
  --config=asan \
  --test_tag_filters=-no_oss,-noasan --build_tag_filters=-no_oss,-noasan \
  --test_output=errors

# TODO(b/178621680): enable ubsan once bazel + clang + ubsan errors are fixed.
#CC=clang readable_run bazel test tensorflow/lite/micro/... --config=ubsan --test_tag_filters=-no_oss,-noubsan --build_tag_filters=-no_oss,-noubsan

CC=clang readable_run bazel test tensorflow/lite/micro/... \
  --test_tag_filters=-no_oss --build_tag_filters=-no_oss \
  --copt=-DTF_LITE_STATIC_MEMORY \
  --test_output=errors

