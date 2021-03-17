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
ROOT_DIR=${SCRIPT_DIR}/../../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

DOWNLOADED_CMSIS_PATH=${DOWNLOADS_DIR}/cmsis

if [ -d ${DOWNLOADED_CMSIS_PATH} ]; then
  echo >&2 "${DOWNLOADED_CMSIS_PATH} already exists, skipping the download."
else

  ZIP_PREFIX="0d7e4fa7131241a17e23dfae18140e0b2e77728f"
  CMSIS_URL="http://github.com/ARM-software/CMSIS_5/archive/${ZIP_PREFIX}.zip"
  CMSIS_MD5="630bb4a0acd3d2f3ccdd8bcccb9d6400"

  # wget is much faster than git clone of the entire repo. So we wget a specific
  # version and can then apply a patch, as needed.
  wget ${CMSIS_URL} -O /tmp/${ZIP_PREFIX}.zip >&2
  check_md5 /tmp/${ZIP_PREFIX}.zip ${CMSIS_MD5}

  unzip -qo /tmp/${ZIP_PREFIX}.zip -d /tmp >&2
  mv /tmp/CMSIS_5-${ZIP_PREFIX} ${DOWNLOADED_CMSIS_PATH}
fi

echo "SUCCESS"
