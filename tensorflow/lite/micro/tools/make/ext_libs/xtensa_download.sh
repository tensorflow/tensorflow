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
# Downloads necessary to build with OPTIMIZED_KERNEL_DIR=xtensa.
#
# Called with four arguments:
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

# Name of the xa_nnlib directory once it is unzipped.
HIFI4_XA_NNLIB_DIRNAME="xa_nnlib_hifi4"

HIFI4_PATH=${DOWNLOADS_DIR}/${HIFI4_XA_NNLIB_DIRNAME}
if [ -d ${HIFI4_PATH} ]; then
  echo >&2 "${HIFI4_PATH} already exists, skipping the download."
else

  ZIP_ARCHIVE_NAME="xa_nnlib_06_27.zip"
  HIFI4_URL="http://mirror.tensorflow.org/github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/${ZIP_ARCHIVE_NAME}"
  HIFI4_MD5="45fdc1209a8da62ab568aa6040f7eabf"

  wget ${HIFI4_URL} -O /tmp/${ZIP_ARCHIVE_NAME} >&2
  MD5=`md5sum /tmp/${ZIP_ARCHIVE_NAME} | awk '{print $1}'`

  if [[ ${MD5} != ${HIFI4_MD5} ]]
  then
    echo "Bad checksum. Expected: ${HIFI4_MD5}, Got: ${MD5}"
    exit 1
  fi

  unzip -qo /tmp/${ZIP_ARCHIVE_NAME} -d ${DOWNLOADS_DIR} >&2
fi

echo "SUCCESS"
