#!/bin/bash 
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

GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core
ARCHS="ARMV7 ARMV7S ARM64 I386 X86_64"

USAGE="usage: compile_ios_tensorflow.sh [-A architecture] [-F cxxflags]

A script to build tensorflow for ios.
This script can only be run on MacOS host platforms.

Options:
-A architecture
Target platforms to compile. The default is: $ARCHS.

-F 
Specify the option flags appending to CXXFLAGS."

while
  ARG="${1-}"
  case "$ARG" in
  -*)  case "$ARG" in -*A*) ARCHS="${2?"$USAGE"}"; shift; esac
       case "$ARG" in -*F*) OPT="${2?"$USAGE"}"; shift; esac
       case "$ARG" in -*[!AF]*) echo "$USAGE" >&2; exit 2;; esac;;
  "")  break;;
  *)   echo "$USAGE" >&2; exit 2;;
  esac
do
  shift
done

for ARCH in ${ARCHS}; do
  ARCH_LOWER=`echo "${ARCH}" | tr '[:upper:]' '[:lower:]'`
  ARCH_LIB=${LIB_PREFIX}-${ARCH_LOWER}.a
  make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
  TARGET=IOS IOS_ARCH=${ARCH} LIB_NAME=${ARCH_LIB} OPTFLAGS="$OPT"
  if [ $? -ne 0 ]
  then
    echo "${ARCH} compilation failed."
    exit 1
  fi
  ARCH_LIBS="${ARCH_LIBS} ${LIBDIR}/ios_${ARCH}/${ARCH_LIB}"
done

lipo \
${ARCH_LIBS} \
-create \
-output ${LIBDIR}/${LIB_PREFIX}.a
