#!/bin/sh
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

# Make sure we're running in Xcode environment
if [ -z "${SRCROOT}" ]
then
      exit 1
fi

# Download TF Lite models from the internet if it does not exist.
FRAMEWORK_FOLDER="${SRCROOT}/Frameworks"
TFLITE_TAR="${FRAMEWORK_FOLDER}/TensorFlowLiteC"
TFLITE_C="${FRAMEWORK_FOLDER}/TensorFlowLiteC-2.4.0"

if [[ -d "$TFLITE_C" ]]; then
  echo "INFO: TFLite frameworks already exist. Skip downloading and use the local frameworks."
else
  mkdir -p "${FRAMEWORK_FOLDER}"
  curl -o "${TFLITE_TAR}" -L "https://dl.google.com/dl/cpdc/e8a95c1d411b795e/TensorFlowLiteC-2.4.0.tar.gz"
  tar -xvf "${TFLITE_TAR}" -C "${FRAMEWORK_FOLDER}"
  rm "${TFLITE_TAR}"
  echo "INFO: Downloaded TensorFlow frameworks to $TFLITE_C."
fi
