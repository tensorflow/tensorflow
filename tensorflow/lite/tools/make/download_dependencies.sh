#!/bin/bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."

DOWNLOADS_DIR=tensorflow/lite/tools/make/downloads
BZL_FILE_PATH=tensorflow/tensorflow.bzl

if [[ "${OSTYPE}" == "darwin"* ]]; then
  function sha256sum() { shasum -a 256 "$@" ; }
fi

# Ensure it is being run from repo root
if [ ! -f $BZL_FILE_PATH ]; then
  echo "Could not find ${BZL_FILE_PATH}":
  echo "Likely you are not running this from the root directory of the repository.";
  exit 1;
fi

EIGEN_WORKSPACE_BZL_PATH="third_party/eigen3/workspace.bzl"
EIGEN_COMMIT="$(grep -oP 'EIGEN_COMMIT = "\K[0-9a-f]{40}' "${EIGEN_WORKSPACE_BZL_PATH}")"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/"${EIGEN_COMMIT}"/eigen-"${EIGEN_COMMIT}".tar.gz"
EIGEN_SHA="$(grep -oP 'EIGEN_SHA256 = "\K[0-9a-f]{64}' "${EIGEN_WORKSPACE_BZL_PATH}")"
GEMMLOWP_WORKSPACE_BZL_PATH="third_party/gemmlowp/workspace.bzl"
GEMMLOWP_COMMIT="$(grep -oP 'GEMMLOWP_COMMIT = "\K[0-9a-f]{40}' "${GEMMLOWP_WORKSPACE_BZL_PATH}")"
GEMMLOWP_URL="https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/"${GEMMLOWP_COMMIT}".zip"
GEMMLOWP_SHA="$(grep -oP 'GEMMLOWP_SHA256 = "\K[0-9a-f]{64}' "${GEMMLOWP_WORKSPACE_BZL_PATH}")"
RUY_URL="https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip"
RUY_SHA="525de68739faa23eeea674596607a3eea7ca4425be2962b26775158e084c1036"
GOOGLETEST_URL="https://github.com/google/googletest/archive/release-1.8.0.tar.gz"
GOOGLETEST_SHA="58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8"
ABSL_WORKSPACE_BZL_PATH="third_party/absl/workspace.bzl"
ABSL_COMMIT="$(grep -oP 'ABSL_COMMIT = "\K[0-9a-f]{40}' "${ABSL_WORKSPACE_BZL_PATH}")"
ABSL_URL="https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/"${ABSL_COMMIT}".tar.gz"
ABSL_SHA="$(grep -oP 'ABSL_SHA256 = "\K[0-9a-f]{64}' "${ABSL_WORKSPACE_BZL_PATH}")"
NEON_2_SSE_URL="https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip"
FARMHASH_WORKSPACE_BZL_PATH="third_party/farmhash/workspace.bzl"
FARMHASH_COMMIT="$(grep -oP 'FARMHASH_COMMIT = "\K[0-9a-f]{40}' "${FARMHASH_WORKSPACE_BZL_PATH}")"
FARMHASH_URL="https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/"${FARMHASH_COMMIT}".tar.gz"
FARMHASH_SHA="$(grep -oP 'FARMHASH_SHA256 = "\K[0-9a-f]{64}' "${FARMHASH_WORKSPACE_BZL_PATH}")"
FLATBUFFERS_URL="https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz"
FLATBUFFERS_SHA="62f2223fb9181d1d6338451375628975775f7522185266cd5296571ac152bc45"
FFT2D_URL="https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz"
FP16_URL="https://github.com/Maratyszcza/FP16/archive/febbb1c163726b5db24bed55cc9dc42529068997.zip"
FFT2D_SHA="ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9"
CPUINFO_URL="https://github.com/pytorch/cpuinfo/archive/c2092219e7c874783a00a62edb94ddc672f57ab3.zip"
CPUINFO_SHA="ea56c399a4f6ca5f749e71acb6a7bfdc653eb65d8f658cb2e414a2fcdca1fe8b"
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
  local usage="Usage: download_and_extract URL DIR [SHA256]"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  local sha256="${3}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  rm -rf ${dir}/*  # Delete existing files.
  tempdir=$(mktemp -d)
  filepath="${tempdir}/$(basename ${url})"
  curl -Lo ${filepath} ${url}
  if [ -n "${sha256}" ]; then
    echo "checking sha256 of ${dir}"
    echo "${sha256}  ${filepath}" | sha256sum -c
  fi
  if [[ "${url}" == *gz ]]; then
    tar -C "${dir}" --strip-components=1 -xzf ${filepath}
  elif [[ "${url}" == *zip ]]; then
    tempdir2=$(mktemp -d)
    unzip ${filepath} -d ${tempdir2}

    # If the zip file contains nested directories, extract the files from the
    # inner directory.
    if ls ${tempdir2}/*/* 1> /dev/null 2>&1; then
      # unzip has no strip components, so unzip to a temp dir, and move the
      # files we want from the tempdir to destination.
      cp -R ${tempdir2}/*/* ${dir}/
    else
      cp -R ${tempdir2}/* ${dir}/
    fi
    rm -rf ${tempdir2}
  fi
  rm -rf ${tempdir}

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${dir}" -type f -name '*BUILD' -delete
}

download_and_extract "${EIGEN_URL}" "${DOWNLOADS_DIR}/eigen" "${EIGEN_SHA}"
download_and_extract "${GEMMLOWP_URL}" "${DOWNLOADS_DIR}/gemmlowp" "${GEMMLOWP_SHA}"
download_and_extract "${RUY_URL}" "${DOWNLOADS_DIR}/ruy" "${RUY_SHA}"
download_and_extract "${GOOGLETEST_URL}" "${DOWNLOADS_DIR}/googletest" "${GOOGLETEST_SHA}"
download_and_extract "${ABSL_URL}" "${DOWNLOADS_DIR}/absl" "${ABSL_SHA}"
download_and_extract "${NEON_2_SSE_URL}" "${DOWNLOADS_DIR}/neon_2_sse"
download_and_extract "${FARMHASH_URL}" "${DOWNLOADS_DIR}/farmhash" "${FARMHASH_SHA}"
download_and_extract "${FLATBUFFERS_URL}" "${DOWNLOADS_DIR}/flatbuffers" "${FLATBUFFERS_SHA}"
download_and_extract "${FFT2D_URL}" "${DOWNLOADS_DIR}/fft2d" "${FFT2D_SHA}"
download_and_extract "${FP16_URL}" "${DOWNLOADS_DIR}/fp16"
download_and_extract "${CPUINFO_URL}" "${DOWNLOADS_DIR}/cpuinfo"

replace_by_sed 's#static uint32x4_t p4ui_CONJ_XOR = vld1q_u32( conj_XOR_DATA );#static uint32x4_t p4ui_CONJ_XOR; // = vld1q_u32( conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint32x2_t p2ui_CONJ_XOR = vld1_u32( conj_XOR_DATA );#static uint32x2_t p2ui_CONJ_XOR;// = vld1_u32( conj_XOR_DATA ); - Removed by scripts#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );#static uint64x2_t p2ul_CONJ_XOR;// = vld1q_u64( p2ul_conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"

echo "download_dependencies.sh completed successfully." >&2
