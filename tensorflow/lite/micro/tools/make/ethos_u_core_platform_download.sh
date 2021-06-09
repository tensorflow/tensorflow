#!/bin/bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH=${DOWNLOADS_DIR}/ethos_u_core_platform

if [ -d ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} ]; then
  echo >&2 "${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} already exists, skipping the download."
else
  UNAME_S=`uname -s`
  if [ ${UNAME_S} == Linux ]; then
    ETHOS_U_CORE_PLATFORM_URL=https://git.mlplatform.org/ml/ethos-u/ethos-u-core-platform.git/snapshot/ethos-u-core-platform-b5f7cfe253dfeadd83caf60fde34b5b66f356782.tar.gz
    EXPECTED_MD5=9431cd98f9d42d3bca9742dd7cab7229
  else
    echo "OS type ${UNAME_S} not supported."
    exit 1
  fi

  TEMPFILE=$(mktemp -d)/temp_file
  wget ${ETHOS_U_CORE_PLATFORM_URL} -O ${TEMPFILE} >&2
  check_md5 ${TEMPFILE} ${EXPECTED_MD5}

  mkdir ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}
  tar xzf ${TEMPFILE} --strip-components=1 -C ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} >&2

  # Run C preprocessor on linker file to get rid of ifdefs and make sure compiler is downloaded first.
  COMPILER=${DOWNLOADS_DIR}/gcc_embedded/bin/arm-none-eabi-gcc
  if [ ! -f ${COMPILER} ]; then
      RETURN_VALUE=`./tensorflow/lite/micro/tools/make/arm_gcc_download.sh ${DOWNLOADS_DIR}`
      if [ "SUCCESS" != "${RETURN_VALUE}" ]; then
        echo "The script ./tensorflow/lite/micro/tools/make/arm_gcc_download.sh failed."
        exit 1
      fi
  fi
  LINKER_PATH=${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}/targets/corstone-300
  ${COMPILER} -E -x c -P -o ${LINKER_PATH}/platform_parsed.ld ${LINKER_PATH}/platform.ld

  # Move rodata from ITCM to DDR in order to support a bigger model without a specified section.
  sed -i '/rodata/d' ${LINKER_PATH}/platform_parsed.ld
  sed -i 's/network_model_sec/\.rodata\*/' ${LINKER_PATH}/platform_parsed.ld

  # Allow tensor_arena in namespace. This will put tensor arena in SRAM intended by linker file.
  sed -i 's/tensor_arena/\*tensor_arena\*/' ${LINKER_PATH}/platform_parsed.ld

  # Patch retarget.c so that g++ can find _exit symbol.
  cat <<EOT >> ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}/targets/corstone-300/retarget.c

void RETARGET(exit)(int return_code) {
  _exit(return_code);
  while (1) {}
}
EOT

fi

echo "SUCCESS"
