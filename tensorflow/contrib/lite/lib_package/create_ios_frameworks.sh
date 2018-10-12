#!/bin/bash -x
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

# TODO(ycling): Refactoring - Move this script into `tools/make`.
set -e

echo "Starting"
TFLITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

TMP_DIR=$(mktemp -d)
echo "Package dir: " $TMP_DIR
FW_DIR=$TMP_DIR/tensorflow_lite_ios_frameworks
FW_DIR_TFLITE=$FW_DIR/tensorflow_lite.framework
FW_DIR_TFLITE_HDRS=$FW_DIR_TFLITE/Headers

echo "Creating target Headers directories"
mkdir -p $FW_DIR_TFLITE_HDRS

echo "Headers, populating: TensorFlow Lite"
cd $TFLITE_DIR/../../..

find tensorflow/contrib/lite -name '*.h' \
    -not -path 'tensorflow/contrib/lite/tools/*' \
    -not -path 'tensorflow/contrib/lite/examples/*' \
    -not -path 'tensorflow/contrib/lite/gen/*' \
    -not -path 'tensorflow/contrib/lite/toco/*' \
    -not -path 'tensorflow/contrib/lite/nnapi/*' \
    -not -path 'tensorflow/contrib/lite/java/*' \
    | tar -cf $FW_DIR_TFLITE_HDRS/tmp.tar -T -
cd $FW_DIR_TFLITE_HDRS
tar xf tmp.tar
rm -f tmp.tar

echo "Headers, populating: Flatbuffer"
cd $TFLITE_DIR/tools/make/downloads/flatbuffers/include/
find . -name '*.h' | tar -cf $FW_DIR_TFLITE_HDRS/tmp.tar -T -
cd $FW_DIR_TFLITE_HDRS
tar xf tmp.tar
rm -f tmp.tar

cd $TFLITE_DIR/../../..
echo "Generate master LICENSE file and copy to target"
bazel build //tensorflow/tools/lib_package:clicenses_generate
cp $TFLITE_DIR/../../../bazel-genfiles/tensorflow/tools/lib_package/include/tensorflow/c/LICENSE \
   $FW_DIR_TFLITE

echo "Copying static libraries"
cp $TFLITE_DIR/tools/make/gen/lib/libtensorflow-lite.a \
   $FW_DIR_TFLITE/tensorflow_lite

# This is required, otherwise they interfere with the documentation of the
# pod at cocoapods.org.
echo "Remove all README files"
cd $FW_DIR_TFLITE_HDRS
find . -type f -name README\* -exec rm -f {} \;
find . -type f -name readme\* -exec rm -f {} \;

TARGET_GEN_LOCATION="$TFLITE_DIR/gen/ios_frameworks"
echo "Moving results to target: " $TARGET_GEN_LOCATION
cd $FW_DIR
zip -q -r tensorflow_lite.framework.zip tensorflow_lite.framework -x .DS_Store
rm -rf $TARGET_GEN_LOCATION
mkdir -p $TARGET_GEN_LOCATION
cp -r tensorflow_lite.framework.zip $TARGET_GEN_LOCATION

echo "Cleaning up"
rm -rf $TMP_DIR

echo "Finished"
