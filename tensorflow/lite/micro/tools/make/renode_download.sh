#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

DOWNLOADED_RENODE_PATH=${DOWNLOADS_DIR}/renode

if [ -d ${DOWNLOADED_RENODE_PATH} ]; then
  echo >&2 "${DOWNLOADED_RENODE_PATH} already exists, skipping the download."
else
  LINUX_PORTABLE_URL="https://github.com/renode/renode/releases/download/v1.11.0/renode-1.11.0.linux-portable.tar.gz"
  TEMP_ARCHIVE="/tmp/renode.tar.gz"

  echo >&2 "Downloading from url: ${LINUX_PORTABLE_URL}"
  wget ${LINUX_PORTABLE_URL} -O ${TEMP_ARCHIVE} >&2

  EXPECTED_MD5="8415361f5caa843f1e31b59c50b2858f"
  check_md5 ${TEMP_ARCHIVE} ${EXPECTED_MD5}

  mkdir ${DOWNLOADED_RENODE_PATH}
  tar xzf ${TEMP_ARCHIVE} --strip-components=1 --directory "${DOWNLOADED_RENODE_PATH}" >&2
  echo >&2 "Unpacked to directory: ${DOWNLOADED_RENODE_PATH}"

  pip3 install -r ${DOWNLOADED_RENODE_PATH}/tests/requirements.txt >&2
fi

echo "SUCCESS"
