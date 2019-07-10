#!/bin/bash -x
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
# Builds the TensorFlow core library with ARM and x86 architectures for iOS, and
# packs them into a fat file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/build_helper.subr"
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

function less_than_required_version() {
  echo $1 | (IFS=. read -r major minor micro
    if [ $major -ne $2 ]; then
      [ $major -lt $2 ]
    elif [ $minor -ne $3 ]; then
      [ $minor -lt $3 ]
    else
      [ ${micro:-0} -lt $4 ]
    fi
  )
}

if [[ -n MACOSX_DEPLOYMENT_TARGET ]]; then
    export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
fi

ACTUAL_XCODE_VERSION=$(xcodebuild -version | head -n 1 | sed 's/Xcode //')
REQUIRED_XCODE_VERSION=7.3.0
if less_than_required_version $ACTUAL_XCODE_VERSION 7 3 0
then
    echo "error: Xcode ${REQUIRED_XCODE_VERSION} or later is required."
    exit 1
fi

usage() {
  echo "Usage: $(basename "$0") [-a]"
  echo "-a [build_arch] build for specified arch comma separate for multiple archs (eg: x86_64,arm64)"
  echo "default is [x86_64, armv7, armv7s, arm64]"
  exit 1
}

BUILD_TARGET="x86_64 armv7 armv7s arm64"
while getopts "a:f:h:n:" opt_name; do
  case "$opt_name" in
    a) BUILD_TARGET="${OPTARG}";;
    f) BUILD_OPT="${OPTARG}";;
    h) NSYNC_HOST="${OPTARG}";;
    n) NSYNC_TARGET="${OPTARG}";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

IFS=' ' read -r -a build_targets <<< "${BUILD_TARGET}"

SCRIPT_DIR=$(cd `dirname $0` && pwd)
source "${SCRIPT_DIR}/build_helper.subr"


GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core

#remove any old artifacts
rm -rf ${LIBDIR}/${LIB_PREFIX}.a

package_tf_library() {
    CAP_DIR=`echo $1 | tr 'a-z' 'A-Z'`
    tf_libs="${LIBDIR}/ios_${CAP_DIR}/${LIB_PREFIX}-${1}.a"
    if [ -f "${LIBDIR}/${LIB_PREFIX}.a" ]; then
        tf_libs="$tf_libs ${LIBDIR}/${LIB_PREFIX}.a"
    fi
    lipo \
    $tf_libs \
    -create \
    -output ${LIBDIR}/${LIB_PREFIX}.a
}

build_tf_target() {
case "$1" in
    armv7)
        make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
        TARGET=IOS IOS_ARCH=ARMV7 LIB_NAME=${LIB_PREFIX}-armv7.a \
        OPTFLAGS="${BUILD_OPT}" HOST_NSYNC_LIB="${NSYNC_HOST}" \
        TARGET_NSYNC_LIB="${NSYNC_TARGET}"
        if [ $? -ne 0 ]
        then
          echo "armv7 compilation failed."
          exit 1
        fi
        package_tf_library "armv7"
        ;;
    armv7s)
        make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
        TARGET=IOS IOS_ARCH=ARMV7S LIB_NAME=${LIB_PREFIX}-armv7s.a \
        OPTFLAGS="${BUILD_OPT}" HOST_NSYNC_LIB="${NSYNC_HOST}" \
        TARGET_NSYNC_LIB="${NSYNC_TARGET}"

        if [ $? -ne 0 ]
        then
          echo "arm7vs compilation failed."
          exit 1
        fi
        package_tf_library "armv7s"
        ;;
    arm64)
        make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
        TARGET=IOS IOS_ARCH=ARM64 LIB_NAME=${LIB_PREFIX}-arm64.a \
        OPTFLAGS="${BUILD_OPT}" HOST_NSYNC_LIB="${NSYNC_HOST}" \
        TARGET_NSYNC_LIB="${NSYNC_TARGET}"
        if [ $? -ne 0 ]
        then
          echo "arm64 compilation failed."
          exit 1
        fi
        package_tf_library "arm64"
        ;;
    x86_64)
        make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
        TARGET=IOS IOS_ARCH=X86_64 LIB_NAME=${LIB_PREFIX}-x86_64.a \
        OPTFLAGS="${BUILD_OPT}" HOST_NSYNC_LIB="${NSYNC_HOST}" \
        TARGET_NSYNC_LIB="${NSYNC_TARGET}"
        if [ $? -ne 0 ]
        then
          echo "x86_64 compilation failed."
          exit 1
        fi
        package_tf_library "x86_64"
        ;;
    *)
        echo "Unknown ARCH"
        exit 1
esac
}

for build_tf_element in "${build_targets[@]}"
do
    echo "$build_tf_element"
    build_tf_target "$build_tf_element"
done

echo "Done building and packaging TF"
file ${LIBDIR}/${LIB_PREFIX}.a
