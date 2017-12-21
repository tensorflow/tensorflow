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
get_cpu_count() {
  case "${OSTYPE}" in
    linux*)
      grep processor /proc/cpuinfo | wc -l ;;
    darwin*)
      sysctl hw.ncpu | awk '{print $2}' ;;
    cygwin*)
      grep processor /proc/cpuinfo | wc -l ;;
    *)
      echo "1"
      exit 1 ;;
  esac
  exit 0
}

get_job_count() {
  echo $(($(get_cpu_count)))
}

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
  echo "Usage: $(basename "$0") [-af]"
  echo "-a [build_arch] build for specified arch comma separate for multiple archs (eg: x86_64,arm64)"
  echo "default is [i386, x86_64, armv7, armv7s, arm64]"
  exit 1
}

BUILD_TARGET="i386 x86_64 armv7 armv7s arm64"
while getopts "a:" opt_name; do
  case "$opt_name" in
    a) BUILD_TARGET="${OPTARG}";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

IFS=' ' read -r -a build_targets <<< "${BUILD_TARGET}"


GENDIR=tensorflow/contrib/lite/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-lite

#remove any old artifacts
rm -rf ${LIBDIR}/${LIB_PREFIX}.a

package_tf_library() {
    tf_libs="${LIBDIR}/ios_${1}/${LIB_PREFIX}-${1}.a"
    if [ -f "${LIBDIR}/${LIB_PREFIX}.a" ]; then
        tf_libs="$tf_libs ${LIBDIR}/${LIB_PREFIX}.a"
    fi
    lipo \
    $tf_libs \
    -create \
    -output ${LIBDIR}/${LIB_PREFIX}.a
}

build_tf_target() {
    make -j"${JOB_COUNT}" -f tensorflow/contrib/lite/Makefile \
    TARGET=IOS IOS_ARCH=${1} LIB_NAME=${LIB_PREFIX}-${1}.a
    if [ $? -ne 0 ]
    then
        echo "${1} compilation failed."
        exit 1
    fi
    package_tf_library "${1}"
}

for build_tf_element in "${build_targets[@]}"
do
    echo "$build_tf_element"
    build_tf_target "$build_tf_element"
done

echo "Done building and packaging TF"
file ${LIBDIR}/${LIB_PREFIX}.a
