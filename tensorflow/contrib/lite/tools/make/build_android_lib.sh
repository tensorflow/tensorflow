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
cd "$SCRIPT_DIR/../../../../.."

# Build using NDK14
# change `TOOLS` for your own env
# `ndkr14` and `toolchain` must under `TOOLS` dir
TOOLS=
if [[ -z $TOOLS ]];then 
  echo "Config your Android Toolchan root dir."
  exit 1
fi 
API=21
export PATH=${TOOLS}/android-toolchain-arm-android-${API}/bin:$PATH
export PATH=${TOOLS}/android-toolchain-arm64-android-${API}/bin:$PATH

# Build library for supported architectures and packs them in a fat binary.
make_library() {
    for arch in armv7 armv8
    do
        if [[ $arch == 'armv7' ]];then
          sysroot=$(TOOLS)/android-toolchain-arm-android-${API}/sysroot 
        else
          sysroot=$(TOOLS)/android-toolchain-arm64-android-${API}/sysroot 
        fi

        API_VERSION=${API} \
        NDK=$(TOOLS) \
        SYSROOT=$sysroot \
        make -f tensorflow/contrib/lite/tools/make/Makefile TARGET=android TARGET_ARCH=${arch} \
        -j 8 micro || exit 1
    done
}

make_library
