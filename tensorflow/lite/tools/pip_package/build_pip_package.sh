#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

set -e

# Find where this script lives and then the Tensorflow root.
MY_DIRECTORY=`dirname $0`
export TENSORFLOW_SRC_ROOT=`realpath $MY_DIRECTORY/../../../..`

export TENSORFLOW_VERSION=`grep "_VERSION = " $TENSORFLOW_SRC_ROOT/tensorflow/tools/pip_package/setup.py  | cut -d'=' -f 2 | sed "s/[ '-]//g"`;


# Build a pip build tree.
BUILD_ROOT=/tmp/tflite_pip
rm -rf $BUILD_ROOT
mkdir -p $BUILD_ROOT/tflite_runtime/lite
mkdir -p $BUILD_ROOT/tflite_runtime/lite/python

# Build an importable module tree
cat > $BUILD_ROOT/tflite_runtime/__init__.py <<EOF;
import tflite_runtime.lite.interpreter
EOF

cat > $BUILD_ROOT/tflite_runtime/lite/__init__.py <<EOF;
from interpreter import Interpreter as Interpreter
EOF

cat > $BUILD_ROOT/tflite_runtime/lite/python/__init__.py <<EOF;
# Python module for TensorFlow Lite
EOF

# Copy necessary source files
TFLITE_ROOT=$TENSORFLOW_SRC_ROOT/tensorflow/lite
cp -r  $TFLITE_ROOT/python/interpreter_wrapper $BUILD_ROOT
cp $TFLITE_ROOT/python/interpreter.py $BUILD_ROOT/tflite_runtime/lite/
cp $TFLITE_ROOT/tools/pip_package/setup.py $BUILD_ROOT
cp $TFLITE_ROOT/tools/pip_package/MANIFEST.in $BUILD_ROOT

# Build the Pip
cd $BUILD_ROOT
python setup.py bdist_wheel
