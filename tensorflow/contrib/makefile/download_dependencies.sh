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

# Ensure it is being run from repo root
if [ ! -f $BZL_FILE_PATH ]; then
  echo "Could not find ${BZL_FILE_PATH}":
  echo "Likely you are not running this from the root directory of the repository.";
  exit 1;
fi

EIGEN_URL="$(grep -o 'http.*bitbucket.org/eigen/eigen/get/.*tar\.gz' "${BZL_FILE_PATH}" | grep -v mirror.bazel | head -n1)"
GEMMLOWP_URL="$(grep -o 'http://mirror.tensorflow.org/github.com/google/gemmlowp/.*zip' "${BZL_FILE_PATH}" | head -n1)"
GOOGLETEST_URL="https://github.com/google/googletest/archive/release-1.8.0.tar.gz"
NSYNC_URL="$(grep -o 'http://mirror.tensorflow.org/github.com/google/nsync/.*tar\.gz' "${BZL_FILE_PATH}" | head -n1)"

# Note: The protobuf repo needs to be cloned due to its submodules.
# These variables contain the GitHub repo and the sha, from `tensorflow/workspace.bzl`,
# from which to clone it from and checkout to.
readonly PROTOBUF_REPO="https://github.com/protocolbuffers/protobuf.git"
readonly PROTOBUF_TAG="$(grep -o 'https://github.com/protocolbuffers/protobuf/archive/.*tar\.gz' "${BZL_FILE_PATH}" | head -n1 | awk '{print substr($0, index($0, "archive") + 8, index($0, "tar") - index($0, "archive") - 9) }')"

# TODO (yongtang): Replace the following with 'http://mirror.tensorflow.org/github.com/google/re2/.*tar\.gz' once
# the archive has been propagated in mirror.tensorflow.org.
RE2_URL="$(grep -o 'https://github.com/google/re2/.*tar\.gz' "${BZL_FILE_PATH}" | head -n1)"
FFT2D_URL="$(grep -o 'http.*fft\.tgz' "${BZL_FILE_PATH}" | grep -v bazel-mirror | head -n1)"
DOUBLE_CONVERSION_URL="$(grep -o "https.*google/double-conversion.*\.zip" "${BZL_FILE_PATH}" | head -n1)"
ABSL_URL="$(grep -o 'https://github.com/abseil/abseil-cpp/.*tar.gz' "${BZL_FILE_PATH}" | head -n1)"
CUB_URL="$(grep -o 'https.*cub/archive.*zip' "${BZL_FILE_PATH}" | grep -v mirror.bazel | head -n1)"

# Required for TensorFlow Lite Flex runtime.
FARMHASH_URL="http://mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz"
FLATBUFFERS_URL="https://github.com/google/flatbuffers/archive/1f5eae5d6a135ff6811724f6c57f911d1f46bb15.tar.gz"

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
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS (AKA darwin) doesn't have wget.
      (cd "${tempdir}"; curl --remote-name --silent --location "${url}")
    else
      wget -P "${tempdir}" "${url}"
    fi
    unzip "${tempdir}"/* -d "${tempdir2}"
    # unzip has no strip components, so unzip to a temp dir, and move the files
    # we want from the tempdir to destination.
    cp -R "${tempdir2}"/*/* "${dir}"/
    rm -rf "${tempdir2}" "${tempdir}"
  fi

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${dir}" -type f -name '*BUILD' -delete
}

function clone_repository() {
  local repo_url="${1}"
  local destination_directory="${2}"
  local commit_sha="${3}"

  if [[ -d "${destination_directory}" ]]; then
    rm -rf "${destination_directory}"
  fi

  git clone "${repo_url}" "${destination_directory}"

  pushd "$(pwd)" 1>/dev/null

  cd "${destination_directory}"

  if [[ -n "${commit_sha}" ]]; then
    git checkout "${PROTOBUF_TAG}"
  fi

  git submodule update --init

  popd 1>/dev/null
}

download_and_extract "${EIGEN_URL}" "${DOWNLOADS_DIR}/eigen"
download_and_extract "${GEMMLOWP_URL}" "${DOWNLOADS_DIR}/gemmlowp"
download_and_extract "${GOOGLETEST_URL}" "${DOWNLOADS_DIR}/googletest"
download_and_extract "${NSYNC_URL}" "${DOWNLOADS_DIR}/nsync"
download_and_extract "${RE2_URL}" "${DOWNLOADS_DIR}/re2"
download_and_extract "${FFT2D_URL}" "${DOWNLOADS_DIR}/fft2d"
download_and_extract "${DOUBLE_CONVERSION_URL}" "${DOWNLOADS_DIR}/double_conversion"
download_and_extract "${ABSL_URL}" "${DOWNLOADS_DIR}/absl"
download_and_extract "${CUB_URL}" "${DOWNLOADS_DIR}/cub/external/cub_archive"

# Required for TensorFlow Lite Flex runtime.
download_and_extract "${FARMHASH_URL}" "${DOWNLOADS_DIR}/farmhash"
download_and_extract "${FLATBUFFERS_URL}" "${DOWNLOADS_DIR}/flatbuffers"

clone_repository "${PROTOBUF_REPO}" "${DOWNLOADS_DIR}/protobuf" "${PROTOBUF_TAG}"

replace_by_sed 's#static uint32x4_t p4ui_CONJ_XOR = vld1q_u32( conj_XOR_DATA );#static uint32x4_t p4ui_CONJ_XOR; // = vld1q_u32( conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint32x2_t p2ui_CONJ_XOR = vld1_u32( conj_XOR_DATA );#static uint32x2_t p2ui_CONJ_XOR;// = vld1_u32( conj_XOR_DATA ); - Removed by scripts#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
replace_by_sed 's#static uint64x2_t p2ul_CONJ_XOR = vld1q_u64( p2ul_conj_XOR_DATA );#static uint64x2_t p2ul_CONJ_XOR;// = vld1q_u64( p2ul_conj_XOR_DATA ); - Removed by script#' \
  "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/arch/NEON/Complex.h"
# TODO(satok): Remove this once protobuf/autogen.sh is fixed.
replace_by_sed 's#https://googlemock.googlecode.com/files/gmock-1.7.0.zip#http://download.tensorflow.org/deps/gmock-1.7.0.zip#' \
  "${DOWNLOADS_DIR}/protobuf/autogen.sh"
cat "third_party/eigen3/gebp_neon.patch" | patch "${DOWNLOADS_DIR}/eigen/Eigen/src/Core/products/GeneralBlockPanelKernel.h"

echo "download_dependencies.sh completed successfully." >&2
