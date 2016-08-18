#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# This is a composite scprit to build all for Android OS

set -e

usage() {
    info "-x use hexagon library located at ../hexagon/<libs and include>"
    exit 1
}

while getopts "x" opt_name; do
    case "$opt_name" in
        x) USE_HEXAGON=true;;
        *) usage;;
    esac
done

if [[ -z "${NDK_ROOT}" ]]; then
    echo "NDK_ROOT should be set as an environment variable" 1>&2
    exit 1
fi

# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}/../../../

# Remove any old files first.
make -f tensorflow/contrib/makefile/Makefile clean
rm -rf tensorflow/contrib/makefile/downloads

# Pull down the required versions of the frameworks we need.
tensorflow/contrib/makefile/download_dependencies.sh

# Compile protobuf for the target Android device architectures.
CC_PREFIX="${CC_PREFIX}" NDK_ROOT="${NDK_ROOT}" \
tensorflow/contrib/makefile/compile_android_protobuf.sh -c

if [[ "${USE_HEXAGON}" == "true" ]]; then
    HEXAGON_PARENT_DIR=$(cd ../hexagon && pwd)
    HEXAGON_LIBS="${HEXAGON_PARENT_DIR}/libs"
    HEXAGON_INCLUDE="${HEXAGON_PARENT_DIR}/include"
fi

make -f tensorflow/contrib/makefile/Makefile \
TARGET=ANDROID NDK_ROOT="${NDK_ROOT}" CC_PREFIX="${CC_PREFIX}" \
HEXAGON_LIBS="${HEXAGON_LIBS}" HEXAGON_INCLUDE="${HEXAGON_INCLUDE}"
