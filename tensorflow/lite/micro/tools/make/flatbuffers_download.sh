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

DOWNLOADED_FLATBUFFERS_PATH=${DOWNLOADS_DIR}/flatbuffers
if [ -d ${DOWNLOADED_FLATBUFFERS_PATH} ]; then
  echo >&2 "${DOWNLOADED_FLATBUFFERS_PATH} already exists, skipping the download."
else
  ZIP_PREFIX="dca12522a9f9e37f126ab925fd385c807ab4f84e"
  FLATBUFFERS_URL="http://mirror.tensorflow.org/github.com/google/flatbuffers/archive/${ZIP_PREFIX}.zip"
  FLATBUFFERS_MD5="aa9adc93eb9b33fa1a2a90969e48baee"

  wget ${FLATBUFFERS_URL} -O /tmp/${ZIP_PREFIX}.zip >&2
  MD5=`md5sum /tmp/${ZIP_PREFIX}.zip | awk '{print $1}'`

  if [[ ${MD5} != ${FLATBUFFERS_MD5} ]]
  then
    echo "Bad checksum. Expected: ${FLATBUFFERS_MD5}, Got: ${MD5}"
    exit 1
  fi

  unzip -qo /tmp/${ZIP_PREFIX}.zip -d /tmp >&2
  mv /tmp/flatbuffers-${ZIP_PREFIX} ${DOWNLOADED_FLATBUFFERS_PATH}
fi

echo "SUCCESS"
