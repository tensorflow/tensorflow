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
cd "$SCRIPT_DIR/../../../../../.."

DOWNLOADS_DIR=tensorflow/lite/experimental/micro/tools/make/downloads
BZL_FILE_PATH=tensorflow/workspace.bzl
AP3_DOWNLOADS_DIR=tensorflow/lite/experimental/micro/examples/micro_speech

# Ensure it is being run from repo root
if [ ! -f $BZL_FILE_PATH ]; then
  echo "Could not find ${BZL_FILE_PATH}":
  echo "Likely you are not running this from the root directory of the repository.";
  exit 1;
fi

GEMMLOWP_URL="https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
FLATBUFFERS_URL="https://github.com/google/flatbuffers/archive/1f5eae5d6a135ff6811724f6c57f911d1f46bb15.tar.gz"
CMSIS_URL="https://github.com/ARM-software/CMSIS_5/archive/5.4.0.zip"
STM32_BARE_LIB_URL="https://github.com/google/stm32_bare_lib/archive/c07d611fb0af58450c5a3e0ab4d52b47f99bc82d.zip"
AP3_URL="https://github.com/AmbiqMicro/TFLiteMicro_Apollo3/archive/dfbcef9a57276c087d95aab7cb234f1d4c9eaaba.zip"
CUST_CMSIS_URL="https://github.com/AmbiqMicro/TFLiteMicro_CustCMSIS/archive/8f63966c5692e6a3a83956efd2e4aed77c4c9949.zip"

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

download_and_extract "${GEMMLOWP_URL}" "${DOWNLOADS_DIR}/gemmlowp"
download_and_extract "${FLATBUFFERS_URL}" "${DOWNLOADS_DIR}/flatbuffers"
download_and_extract "${CMSIS_URL}" "${DOWNLOADS_DIR}/cmsis"
download_and_extract "${STM32_BARE_LIB_URL}" "${DOWNLOADS_DIR}/stm32_bare_lib"
download_and_extract "${AP3_URL}" "${AP3_DOWNLOADS_DIR}/apollo3_ext"
download_and_extract "${CUST_CMSIS_URL}" "${AP3_DOWNLOADS_DIR}/CMSIS_ext"

echo "download_dependencies.sh completed successfully." >&2
