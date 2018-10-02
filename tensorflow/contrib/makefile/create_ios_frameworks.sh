#! /bin/sh

#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# Must be run after: build_all_ios.sh
# Creates an iOS framework which is placed under:
#    gen/ios_frameworks/tensorflow_experimental.framework.zip

set -e
pushd .

echo "Starting"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMP_DIR=$(mktemp -d)
echo "Package dir: " $TMP_DIR
FW_DIR=$TMP_DIR/tensorflow_ios_frameworks
FW_DIR_TFCORE=$FW_DIR/tensorflow_experimental.framework
FW_DIR_TFCORE_HDRS=$FW_DIR_TFCORE/Headers

echo "Creating target Headers directories"
mkdir -p $FW_DIR_TFCORE_HDRS

echo "Generate master LICENSE file and copy to target"
bazel build //tensorflow/tools/lib_package:clicenses_generate
cp $SCRIPT_DIR/../../../bazel-genfiles/tensorflow/tools/lib_package/include/tensorflow/c/LICENSE \
   $FW_DIR_TFCORE

echo "Copying static libraries"
cp $SCRIPT_DIR/gen/lib/libtensorflow-core.a \
   $FW_DIR_TFCORE/tensorflow_experimental
cp $SCRIPT_DIR/gen/protobuf_ios/lib/libprotobuf.a \
   $FW_DIR_TFCORE/libprotobuf_experimental.a

echo "Headers, populating: tensorflow (core)"
cd $SCRIPT_DIR/../../..
find tensorflow -name "*.h" | tar -cf $FW_DIR_TFCORE_HDRS/tmp.tar -T -
cd $FW_DIR_TFCORE_HDRS
tar xf tmp.tar
rm -f tmp.tar

echo "Headers, populating: third_party"
cd $SCRIPT_DIR/../../..
tar cf $FW_DIR_TFCORE_HDRS/tmp.tar third_party
cd $FW_DIR_TFCORE_HDRS
tar xf tmp.tar
rm -f tmp.tar

echo "Headers, populating: unsupported"
cd $SCRIPT_DIR/downloads/eigen
tar cf $FW_DIR_TFCORE_HDRS/third_party/eigen3/tmp.tar unsupported
cd $FW_DIR_TFCORE_HDRS/third_party/eigen3
tar xf tmp.tar
rm -f tmp.tar

echo "Headers, populating: Eigen"
cd $SCRIPT_DIR/downloads/eigen
tar cf $FW_DIR_TFCORE_HDRS/third_party/eigen3/tmp.tar Eigen
cd $FW_DIR_TFCORE_HDRS/third_party/eigen3
tar xf tmp.tar
rm -f tmp.tar

echo "Headers, populating: tensorflow (protos)"
cd $SCRIPT_DIR/gen/proto
tar cf $FW_DIR_TFCORE_HDRS/tmp.tar tensorflow
cd $FW_DIR_TFCORE_HDRS
tar xf tmp.tar
# Don't include the auto downloaded/generated to build this library
rm -rf tensorflow/contrib/makefile
rm -f tmp.tar

echo "Headers, populating: google (proto src)"
cd $SCRIPT_DIR/downloads/protobuf/src
tar cf $FW_DIR_TFCORE_HDRS/tmp.tar google
cd $FW_DIR_TFCORE_HDRS
tar xf tmp.tar
rm -f tmp.tar

# This is required, otherwise they interfere with the documentation of the
# pod at cocoapods.org
echo "Remove all README files"
cd $FW_DIR_TFCORE_HDRS
find . -type f -name README\* -exec rm -f {} \;
find . -type f -name readme\* -exec rm -f {} \;

TARGET_GEN_LOCATION="$SCRIPT_DIR/gen/ios_frameworks"
echo "Moving results to target: " $TARGET_GEN_LOCATION
cd $FW_DIR
zip -q -r tensorflow_experimental.framework.zip tensorflow_experimental.framework -x .DS_Store
rm -rf $TARGET_GEN_LOCATION
mkdir -p $TARGET_GEN_LOCATION
cp -r tensorflow_experimental.framework.zip $TARGET_GEN_LOCATION

echo "Cleaning up"
popd
rm -rf $TMP_DIR

echo "Finished"
