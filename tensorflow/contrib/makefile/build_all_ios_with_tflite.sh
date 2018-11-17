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

# This shell script is used to build TensorFlow Lite Flex runtime for iOS.
# It compiles TensorFlow Lite and TensorFlow codebases together, and enable a
# route to use TensorFlow kernels in TensorFlow Lite.
#
# After the script is executed, the multi-architecture static libraries will be
# created under: `tensorflow/contrib/makefile/gen/lib/`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_SRCDIR="${SCRIPT_DIR}/../../../"
cd ${TOP_SRCDIR}

# Exporting `WITH_TFLITE_FLEX`. The flag will be propagated all the way
# down to Makefile.
export WITH_TFLITE_FLEX="true"
# Execute `build_all_ios.sh` and propagate all parameters.
tensorflow/contrib/makefile/build_all_ios.sh $*

# Copy all the libraries required for TFLite Flex runtime together.
cd "${TOP_SRCDIR}/tensorflow/contrib/makefile"
cp 'downloads/nsync/builds/lipo.ios.c++11/nsync.a' 'gen/lib/'
cp 'gen/protobuf_ios/lib/libprotobuf.a' 'gen/lib/'
cp 'gen/lib/libtensorflow-core.a' 'gen/lib/libtensorflow-lite.a'
