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
cd "$SCRIPT_DIR/../../.."

DOWNLOADS_DIR=tensorflow/contrib/lite/downloads
BZL_FILE_PATH=tensorflow/workspace.bzl

# Ensure it is being run from repo root
if [ ! -f $BZL_FILE_PATH ]; then
  echo "Could not find ${BZL_FILE_PATH}":
  echo "Likely you are not running this from the root directory of the repository.";
  exit 1;
fi

EIGEN_URL="$(grep -o 'http.*bitbucket.org/eigen/eigen/get/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v mirror.bazel | head -n1)"
GEMMLOWP_URL="$(grep -o 'https://mirror.bazel.build/github.com/google/gemmlowp/.*zip' "${BZL_FILE_PATH}" | head -n1)"
GOOGLETEST_URL="https://github.com/google/googletest/archive/release-1.8.0.tar.gz"
ABSL_URL="$(grep -o 'https://github.com/abseil/abseil-cpp/.*tar.gz' "${BZL_FILE_PATH}" | head -n1)"
NEON_2_SSE_URL="https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip"
FARMHASH_URL="https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz"
FLATBUFFERS_URL="https://github.com/google/flatbuffers/archive/master.zip"
MODELS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_ios_lite_float_2017_11_08.zip"
QUANTIZED_MODELS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip"

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
  if [[ "${url}" == *gz ]]; then
    curl -Ls "${url}" | tar -C "${dir}" --strip-components=1 -xz
  elif [[ "${url}" == *zip ]]; then
    tempdir=$(mktemp -d)
    tempdir2=$(mktemp -d)

    curl -L ${url} > ${tempdir}/zipped.zip
    unzip ${tempdir}/zipped.zip -d ${tempdir2}

    # If the zip file contains nested directories, extract the files from the
    # inner directory.
    if ls ${tempdir2}/*/* 1> /dev/null 2>&1; then
      # unzip has no strip components, so unzip to a temp dir, and move the
      # files we want from the tempdir to destination.
      cp -R ${tempdir2}/*/* ${dir}/
    else
      cp -R ${tempdir2}/* ${dir}/
    fi
    rm -rf ${tempdir2} ${tempdir}
  fi

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${dir}" -type f -name '*BUILD' -delete
}

download_and_extract "${EIGEN_URL}" "${DOWNLOADS_DIR}/eigen"
download_and_extract "${GEMMLOWP_URL}" "${DOWNLOADS_DIR}/gemmlowp"
download_and_extract "${GOOGLETEST_URL}" "${DOWNLOADS_DIR}/googletest"
download_and_extract "${ABSL_URL}" "${DOWNLOADS_DIR}/absl"
download_and_extract "${NEON_2_SSE_URL}" "${DOWNLOADS_DIR}/neon_2_sse"
download_and_extract "${FARMHASH_URL}" "${DOWNLOADS_DIR}/farmhash"
download_and_extract "${FLATBUFFERS_URL}" "${DOWNLOADS_DIR}/flatbuffers"
download_and_extract "${MODELS_URL}" "${DOWNLOADS_DIR}/models"
download_and_extract "${QUANTIZED_MODELS_URL}" "${DOWNLOADS_DIR}/quantized_models"

replace_by_sed 's#static uint32x4_t p4ui_CONJ_XOR = vld1q_u32( conj_XOR_DATA );#static uint32x4_t p4ui_CONJ_XOR; // = vld1q_u32( conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint32x2_t p2ui_CONJ_XOR = vld1_u32( conj_XOR_DATA );#static uint32x2_t p2ui_CONJ_XOR;// = vld1_u32( conj_XOR_DATA ); - Removed by scripts#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );#static uint64x2_t p2ul_CONJ_XOR;// = vld1q_u64( p2ul_conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"

cp ${DOWNLOADS_DIR}/models/models/* tensorflow/contrib/lite/examples/ios/simple/data/
cp ${DOWNLOADS_DIR}/quantized_models/* tensorflow/contrib/lite/examples/ios/camera/data/

echo "download_dependencies.sh completed successfully." >&2
