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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."

profiling_args=

while getopts "p" opt; do
  case $opt in
    p) profiling_args='-DGEMMLOWP_PROFILING,-DTFLITE_PROFILING_ENABLED';;
    *) printf "if you want to enable profiling: pass in [-p]\n"
      exit 2;;
  esac
done

shift $(($OPTIND - 1))
# Build library for supported architectures and packs them in a fat binary.
make_library() {
    for arch in x86_64 armv7 armv7s arm64
    do
        make -f tensorflow/lite/tools/make/Makefile TARGET=ios TARGET_ARCH=${arch} EXTRA_CXXFLAGS=$profiling_args \
        -j 8
    done
    mkdir -p tensorflow/lite/tools/make/gen/lib
    lipo \
    tensorflow/lite/tools/make/gen/ios_x86_64/lib/${1} \
    tensorflow/lite/tools/make/gen/ios_armv7/lib/${1} \
    tensorflow/lite/tools/make/gen/ios_armv7s/lib/${1} \
    tensorflow/lite/tools/make/gen/ios_arm64/lib/${1} \
    -create \
    -output tensorflow/lite/tools/make/gen/lib/${1}
}

make_library libtensorflow-lite.a
make_library benchmark-lib.a
