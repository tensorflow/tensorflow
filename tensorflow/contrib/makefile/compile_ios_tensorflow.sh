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
  echo $1 | (IFS=. read major minor micro
    if [ $major -ne $2 ]; then
      [ $major -lt $2 ]
    elif [ $minor -ne $3 ]; then
      [ $minor -lt $3 ]
    else
      [ ${micro:-0} -lt $4 ]
    fi
  )
}

ACTUAL_XCODE_VERSION=`xcodebuild -version | head -n 1 | sed 's/Xcode //'`
REQUIRED_XCODE_VERSION=7.3.0
if less_than_required_version $ACTUAL_XCODE_VERSION 7 3 0
then
    echo "error: Xcode ${REQUIRED_XCODE_VERSION} or later is required."
    exit 1
fi

GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core

make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7 LIB_NAME=${LIB_PREFIX}-armv7.a OPTFLAGS="$1" 
if [ $? -ne 0 ]
then
  echo "armv7 compilation failed."
  exit 1
fi

make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7S LIB_NAME=${LIB_PREFIX}-armv7s.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "arm7vs compilation failed."
  exit 1
fi

make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARM64 LIB_NAME=${LIB_PREFIX}-arm64.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "arm64 compilation failed."
  exit 1
fi

make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=I386 LIB_NAME=${LIB_PREFIX}-i386.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "i386 compilation failed."
  exit 1
fi

make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=X86_64 LIB_NAME=${LIB_PREFIX}-x86_64.a OPTFLAGS="$1"
if [ $? -ne 0 ]
then
  echo "x86_64 compilation failed."
  exit 1
fi

lipo \
${LIBDIR}/ios_ARMV7/${LIB_PREFIX}-armv7.a \
${LIBDIR}/ios_ARMV7S/${LIB_PREFIX}-armv7s.a \
${LIBDIR}/ios_ARM64/${LIB_PREFIX}-arm64.a \
${LIBDIR}/ios_I386/${LIB_PREFIX}-i386.a \
${LIBDIR}/ios_X86_64/${LIB_PREFIX}-x86_64.a \
-create \
-output ${LIBDIR}/${LIB_PREFIX}.a
