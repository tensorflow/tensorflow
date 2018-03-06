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

make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=x86_64 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=i386 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=armv7 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=armv7s -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=arm64 -j 8

lipo \
tensorflow/contrib/lite/gen/lib/ios_x86_64/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_i386/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_armv7/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_armv7s/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_arm64/libtensorflow-lite.a \
-create \
-output tensorflow/contrib/lite/gen/lib/libtensorflow-lite.a
