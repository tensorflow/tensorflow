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

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_ios_lite_float_2017_11_08.zip"
QUANTIZED_MODELS_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip"
DOWNLOADS_DIR=$(mktemp -d)

cd $SCRIPT_DIR

download_and_extract() {
  local usage="Usage: download_and_extract URL DIR"
  local url="${1:?${usage}}"
  local dir="${2:?${usage}}"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
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
}

download_and_extract "${MODELS_URL}" "${DOWNLOADS_DIR}/models"
download_and_extract "${QUANTIZED_MODELS_URL}" "${DOWNLOADS_DIR}/quantized_models"

file ${DOWNLOADS_DIR}/models

cp ${DOWNLOADS_DIR}/models/models/* simple/data/
cp ${DOWNLOADS_DIR}/models/models/* camera/data/
cp "${DOWNLOADS_DIR}/quantized_models/mobilenet_quant_v1_224.tflite" \
   'camera/data/mobilenet_quant_v1_224.tflite'
