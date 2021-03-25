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

# This script will download and setup the riscv-linux required toolchain and qemu tool.

set -e
set -o pipefail

BOOTSTRAP_SCRIPT_PATH=$(dirname "$0")
BOOTSTRAP_WORK_DIR=${BOOTSTRAP_SCRIPT_PATH}/.bootstrap

PREBUILT_DIR=${BOOTSTRAP_SCRIPT_PATH}/Prebuilt

read -p "Enter the riscv tools root path(press enter to use default path:${PREBUILT_DIR}): " INPUT_PATH
if [[ "${INPUT_PATH}" ]]; then
  PREBUILT_DIR=${INPUT_PATH}
fi
echo "The riscv tool prefix path: ${PREBUILT_DIR}"

if [[ "${OSTYPE}" == "linux-gnu" ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=13q6sYVlae-hRrgj7SNlvbJFYI3Q8sCI4
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=rvv-llvm-toolchain.tar.bz2
  QEMU_FILE_ID=1JkLana7CGeD2wfwn8shHQxcYrv3j1l9s
  QEMU_FILE_NAME=riscv-qemu-v5.2.0-rvv-rvb-zfh-856da0e-linux-ubuntu.tar.gz

  TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}/toolchain/clang/linux/RISCV
  QEMU_PATH_PREFIX=${PREBUILT_DIR}/qemu/linux/RISCV
elif [[ "${OSTYPE}" == "darwin"* ]]; then
  RISCV_CLANG_TOOLCHAIN_FILE_ID=empty
  RISCV_CLANG_TOOLCHAIN_FILE_NAME=empty
  QEMU_FILE_ID=empty
  QEMU_FILE_NAME=empty

  TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}/toolchain/clang/darwin/RISCV
  QEMU_PATH_PREFIX=${PREBUILT_DIR}/qemu/darwin/RISCV

  echo "We haven't had the darwin prebuilt binary yet. Skip this script."
  exit 1
else
  echo "${OSTYPE} is not supported."
  exit 1
fi

function cleanup {
  if [[ -d ${BOOTSTRAP_WORK_DIR} ]]; then
    rm -rf ${BOOTSTRAP_WORK_DIR}
  fi
}

# Call the cleanup function when this tool exits.
trap cleanup EXIT

wget_google_drive() {
  local file_id="$1"
  local file_name="$2"
  local install_path="$3"
  local tar_option="$4"

  wget --save-cookies ${BOOTSTRAP_WORK_DIR}/cookies.txt \
    "https://docs.google.com/uc?export=download&id="$file_id -O- | \
    sed -En "s/.*confirm=([0-9A-Za-z_]+).*/\1/p" > ${BOOTSTRAP_WORK_DIR}/confirm.txt
  wget --progress=bar:force:noscroll --load-cookies ${BOOTSTRAP_WORK_DIR}/cookies.txt \
    "https://docs.google.com/uc?export=download&id=$file_id&confirm=`cat ${BOOTSTRAP_WORK_DIR}/confirm.txt`" -O- | \
    tar $tar_option - --no-same-owner --strip-components=1 -C $install_path
}

download_file() {
  # server name or google drive file_id
  local file_download_info="$1"
  local file_name="$2"
  local install_path="$3"
  # download method(e.g. wget_google_drive)
  local download_method="$4"

  echo "Install $2 to $3"
  if [[ -e $3/file_info.txt ]]; then
    read -p "The file already exists. Keep it (y/n)? " replaced
    case ${replaced:0:1} in
      y|Y )
        echo "Skip download $2."
        return
      ;;
      * )
        rm -rf $3
      ;;
    esac
  fi

  local tar_option=""
  if [[ "${file_name##*.}" == "gz" ]]; then
    tar_option="zxpf"
  elif [[ "${file_name##*.}" == "bz2" ]]; then
    tar_option="jxpf"
  fi
  echo "tar option: $tar_option"

  echo "Download $file_name ..."
  mkdir -p $install_path
  $download_method $file_download_info $file_name $install_path $tar_option

  echo "$file_download_info $file_name" > $install_path/file_info.txt
}

mkdir -p ${BOOTSTRAP_WORK_DIR}

read -p "Install RISCV clang toolchain(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file ${RISCV_CLANG_TOOLCHAIN_FILE_ID} \
                  ${RISCV_CLANG_TOOLCHAIN_FILE_NAME} \
                  ${TOOLCHAIN_PATH_PREFIX} \
                  wget_google_drive
  ;;
  * )
    echo "Skip RISCV clang toolchain."
  ;;
esac

read -p "Install RISCV qemu(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file $QEMU_FILE_ID \
                  ${QEMU_FILE_NAME} \
                  ${QEMU_PATH_PREFIX} \
                  wget_google_drive
  ;;
  * )
    echo "Skip RISCV qemu."
  ;;
esac

echo "Bootstrap finished."
