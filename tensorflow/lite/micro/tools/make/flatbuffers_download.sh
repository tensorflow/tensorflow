#!/bin/bash
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
#
# Called with following arguments:
# 1 - Path to the downloads folder which is typically
#     tensorflow/lite/micro/tools/make/downloads
#
# This script is called from the Makefile and uses the following convention to
# enable determination of sucess/failure:
#
#   - If the script is successful, the only output on stdout should be SUCCESS.
#     The makefile checks for this particular string.
#
#   - Any string on stdout that is not SUCCESS will be shown in the makefile as
#     the cause for the script to have failed.
#
#   - Any other informational prints should be on stderr.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

# TODO(b/173239141): Patch flatbuffers to avoid pulling in extra symbols from
# strtod that are not used at runtime but are still problematic on the
# Bluepill platform.
#
# Parameter(s):
#   $1 - full path to the downloaded flexbuffers.h that will be patched in-place.
function patch_to_avoid_strtod() {
  local input_flexbuffers_path="$1"
  local temp_flexbuffers_path="/tmp/flexbuffers_patched.h"
  local string_to_num_line=`awk '/StringToNumber/{ print NR; }' ${input_flexbuffers_path}`
  local case_string_line=$((${string_to_num_line} - 2))

  head -n ${case_string_line} ${input_flexbuffers_path} > ${temp_flexbuffers_path}

  echo "#if 1" >> ${temp_flexbuffers_path}
  echo "#pragma GCC diagnostic push" >> ${temp_flexbuffers_path}
  echo "#pragma GCC diagnostic ignored \"-Wnull-dereference\"" >> ${temp_flexbuffers_path}
  echo "          // TODO(b/173239141): Patched via micro/tools/make/flexbuffers_download.sh" >> ${temp_flexbuffers_path}
  echo "          // Introduce a segfault for an unsupported code path for TFLM." >> ${temp_flexbuffers_path}
  echo "          return *(static_cast<double*>(nullptr));" >> ${temp_flexbuffers_path}
  echo "#pragma GCC diagnostic pop" >> ${temp_flexbuffers_path}
  echo "#else" >> ${temp_flexbuffers_path}
  echo "          // This is the original code" >> ${temp_flexbuffers_path}
  sed -n -e $((${string_to_num_line} -  1)),$((${string_to_num_line} + 1))p ${input_flexbuffers_path} >> ${temp_flexbuffers_path}
  echo "#endif" >> ${temp_flexbuffers_path}

  local total_num_lines=`wc -l ${input_flexbuffers_path} | awk '{print $1}'`
  sed -n -e $((${string_to_num_line} + 2)),${total_num_lines}p ${input_flexbuffers_path} >> ${temp_flexbuffers_path}
  mv ${input_flexbuffers_path} ${input_flexbuffers_path}.orig
  mv ${temp_flexbuffers_path} ${input_flexbuffers_path}
}

# The BUILD files in the downloaded folder result in an error with:
#  bazel build tensorflow/lite/micro/...
#
# Parameters:
#   $1 - path to the downloaded flatbuffers code.
function delete_build_files() {
  rm -f `find ${1} -name BUILD`
  rm -f `find ${1} -name BUILD.bazel`
}

DOWNLOADED_FLATBUFFERS_PATH=${DOWNLOADS_DIR}/flatbuffers

if [ -d ${DOWNLOADED_FLATBUFFERS_PATH} ]; then
  echo >&2 "${DOWNLOADED_FLATBUFFERS_PATH} already exists, skipping the download."
else
  ZIP_PREFIX="dca12522a9f9e37f126ab925fd385c807ab4f84e"
  FLATBUFFERS_URL="http://mirror.tensorflow.org/github.com/google/flatbuffers/archive/${ZIP_PREFIX}.zip"
  FLATBUFFERS_MD5="aa9adc93eb9b33fa1a2a90969e48baee"

  wget ${FLATBUFFERS_URL} -O /tmp/${ZIP_PREFIX}.zip >&2
  check_md5 /tmp/${ZIP_PREFIX}.zip ${FLATBUFFERS_MD5}

  unzip -qo /tmp/${ZIP_PREFIX}.zip -d /tmp >&2
  mv /tmp/flatbuffers-${ZIP_PREFIX} ${DOWNLOADED_FLATBUFFERS_PATH}

  patch_to_avoid_strtod ${DOWNLOADED_FLATBUFFERS_PATH}/include/flatbuffers/flexbuffers.h
  delete_build_files ${DOWNLOADED_FLATBUFFERS_PATH}
fi

echo "SUCCESS"
