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

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

# The BUILD files in the downloaded folder result in an error with:
#  bazel build tensorflow/lite/micro/...
#
# Parameters:
#   $1 - path to the downloaded flatbuffers code.
function delete_build_files() {
  rm -f `find ${1} -name BUILD`
}

DOWNLOADED_PIGWEED_PATH=${DOWNLOADS_DIR}/pigweed

if [ -d ${DOWNLOADED_PIGWEED_PATH} ]; then
  echo >&2 "${DOWNLOADED_PIGWEED_PATH} already exists, skipping the download."
else
  git clone https://pigweed.googlesource.com/pigweed/pigweed ${DOWNLOADED_PIGWEED_PATH} >&2
  pushd ${DOWNLOADED_PIGWEED_PATH} > /dev/null
  git checkout 47268dff45019863e20438ca3746c6c62df6ef09 >&2

  # Patch for TFLM specific changes that are not currently upstreamed.
  git apply ../../pigweed.patch
  popd > /dev/null

  delete_build_files ${DOWNLOADED_PIGWEED_PATH}
fi

echo "SUCCESS"
