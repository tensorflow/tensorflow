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
FLOAT_MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz"
QUANTIZED_MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
DOWNLOADS_DIR=$(mktemp -d)

cd "$SCRIPT_DIR"

download_and_extract() {
  local url="$1"
  local dir="$2"
  echo "downloading ${url}" >&2
  mkdir -p "${dir}"
  tempdir=$(mktemp -d)

  curl -L ${url} > ${tempdir}/archive.tgz
  cd ${dir}
  tar zxvf ${tempdir}/archive.tgz
  rm -rf ${tempdir}
}

download_and_extract "${FLOAT_MODEL_URL}" "${DOWNLOADS_DIR}/float_model"
download_and_extract "${QUANTIZED_MODEL_URL}" "${DOWNLOADS_DIR}/quantized_model"

cd "$SCRIPT_DIR"
cp "${DOWNLOADS_DIR}/float_model/mobilenet_v1_1.0_224.tflite" "simple/data/mobilenet_v1_1.0_224.tflite"
cp "${DOWNLOADS_DIR}/float_model/mobilenet_v1_1.0_224.tflite" "camera/data/mobilenet_v1_1.0_224.tflite"
cp "${DOWNLOADS_DIR}/quantized_model/mobilenet_v1_1.0_224_quant.tflite" \
   'camera/data/mobilenet_quant_v1_224.tflite'
echo "Done"
