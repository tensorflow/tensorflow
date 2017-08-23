#!/bin/bash
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

set -e

DOWNLOADS_DIR=tensorflow/contrib/makefile/downloads
BZL_FILE_PATH=tensorflow/workspace.bzl

EIGEN_URL="$(grep -o 'http.*bitbucket.org/eigen/eigen/get/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
GEMMLOWP_URL="$(grep -o 'http.*github.com/google/gemmlowp/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
GOOGLETEST_URL="https://github.com/google/googletest/archive/release-1.8.0.tar.gz"
NSYNC_URL="$(grep -o 'http.*github.com/google/nsync/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
PROTOBUF_URL="$(grep -o 'http.*github.com/google/protobuf/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
RE2_URL="$(grep -o 'http.*github.com/google/re2/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
FFT2D_URL="$(grep -o 'http.*fft\.tgz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"

# TODO(petewarden): Some new code in Eigen triggers a clang bug with iOS arm64,
#                   so work around it by patching the source.
replace_by_sed() {
  local regex="${1}"
  shift
  # Detect the version of sed by the return value of "--version" flag. GNU-sed
  # supports "--version" while BSD-sed doesn't.
  if ! sed --version >/dev/null 2>&1; then
    # BSD-sed.
    sed -i '' -e "${regex}" "$@"
  else
    # GNU-sed.
    sed -i -e "${regex}" "$@"
  fi
}

download_and_extract() {
  local usage="Usage: download_and_extract URL DIR"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  curl -Ls "${url}" | tar -C "${dir}" --strip-components=1 -xz

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${dir}" -type f -name '*BUILD' -delete
}

download_and_extract "${EIGEN_URL}" "${DOWNLOADS_DIR}/eigen"
download_and_extract "${GEMMLOWP_URL}" "${DOWNLOADS_DIR}/gemmlowp"
download_and_extract "${GOOGLETEST_URL}" "${DOWNLOADS_DIR}/googletest"
download_and_extract "${NSYNC_URL}" "${DOWNLOADS_DIR}/nsync"
download_and_extract "${PROTOBUF_URL}" "${DOWNLOADS_DIR}/protobuf"
download_and_extract "${RE2_URL}" "${DOWNLOADS_DIR}/re2"
download_and_extract "${FFT2D_URL}" "${DOWNLOADS_DIR}/fft2d"

replace_by_sed 's#static uint32x4_t p4ui_CONJ_XOR = vld1q_u32( conj_XOR_DATA );#static uint32x4_t p4ui_CONJ_XOR; // = vld1q_u32( conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint32x2_t p2ui_CONJ_XOR = vld1_u32( conj_XOR_DATA );#static uint32x2_t p2ui_CONJ_XOR;// = vld1_u32( conj_XOR_DATA ); - Removed by scripts#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );#static uint64x2_t p2ul_CONJ_XOR;// = vld1q_u64( p2ul_conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
# TODO(satok): Remove this once protobuf/autogen.sh is fixed.
replace_by_sed 's#https://googlemock.googlecode.com/files/gmock-1.7.0.zip#http://download.tensorflow.org/deps/gmock-1.7.0.zip#' \
  "${DOWNLOADS_DIR}/protobuf/autogen.sh"

echo "download_dependencies.sh completed successfully." >&2
