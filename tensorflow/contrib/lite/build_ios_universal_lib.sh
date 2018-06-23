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
cd "$SCRIPT_DIR/../../.."

# Build library for supported architectures and packs them in a fat binary.
make_library() {
    for arch in x86_64 i386 armv7 armv7s arm64
    do
        make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=${arch} \
        -j 8 \
        $SCRIPT_DIR/gen/lib/ios_${arch}/${1}
    done
    lipo \
    tensorflow/contrib/lite/gen/lib/ios_x86_64/${1} \
    tensorflow/contrib/lite/gen/lib/ios_i386/${1} \
    tensorflow/contrib/lite/gen/lib/ios_armv7/${1} \
    tensorflow/contrib/lite/gen/lib/ios_armv7s/${1} \
    tensorflow/contrib/lite/gen/lib/ios_arm64/${1} \
    -create \
    -output tensorflow/contrib/lite/gen/lib/${1}
}

make_library libtensorflow-lite.a
make_library benchmark-lib.a
