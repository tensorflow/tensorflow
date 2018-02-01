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
# This is a composite script to build all for Android OS

set -e

usage() {
  echo "Usage: NDK_ROOT=<path to ndk root> $(basename "$0") [-Es:t:Tx:a]"
  echo "-E enable experimental hexnn ops"
  echo "-s [sub_makefiles] sub makefiles separated by white space"
  echo "-t [build_target] build target for Android makefile [default=all]"
  echo "-T only build tensorflow"
  echo "-x [hexagon library path] copy and hexagon libraries in the specified path"
  echo "-a [architecture] Architecture of target android [default=armeabi-v7a] \
(supported architecture list: \
arm64-v8a armeabi armeabi-v7a mips mips64 x86 x86_64 tegra)"
  exit 1
}

if [[ -z "${NDK_ROOT}" ]]; then
    echo "NDK_ROOT should be set as an environment variable" 1>&2
    exit 1
fi

ARCH=armeabi-v7a

while getopts "Es:t:Tx:a" opt_name; do
  case "$opt_name" in
    E) ENABLE_EXPERIMENTAL_HEXNN_OPS="true";;
    s) SUB_MAKEFILES="${OPTARG}";;
    t) BUILD_TARGET="${OPTARG}";;
    T) ONLY_MAKE_TENSORFLOW="true";;
    x) HEXAGON_LIB_PATH="${OPTARG}";;
    a) ARCH="${OPTARG}";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

if [ "$ARCH" == "tegra" ]; then
    if [[ -z "${JETPACK}" ]]; then
        export JETPACK="$HOME/JetPack_Android_3.0"
    fi
    if [ ! -d ${JETPACK} ]; then
        echo "Can't find Jetpack at ${JETPACK}"
        echo "Set JETPACK=<path to Jetpack Android> to specify a non-default Jetpack path"
        exit -1
    fi
    if [ ! -d ${JETPACK}/cuda ]; then
        ln -s $(ls -d ${JETPACK}/cuda-*/|sort -r|head -n1) ${JETPACK}/cuda
    fi
    if [ ! -d ${JETPACK}/cuda ]; then
        ln -s $(ls -d ${JETPACK}/cuda-*/|sort -r|head -n1) ${JETPACK}/cuda
    fi

    export BUILD_FOR_TEGRA=1
    ARCH="arm64-v8a"
fi

# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${SCRIPT_DIR}"/../../../

source "${SCRIPT_DIR}/build_helper.subr"
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

HEXAGON_DOWNLOAD_PATH="tensorflow/contrib/makefile/downloads/hexagon"

# Remove any old files first.
make -f tensorflow/contrib/makefile/Makefile cleantarget

if [[ "${ONLY_MAKE_TENSORFLOW}" != "true" ]]; then
  rm -rf tensorflow/contrib/makefile/downloads
  # Pull down the required versions of the frameworks we need.
  tensorflow/contrib/makefile/download_dependencies.sh
  # Compile protobuf for the target Android device architectures.
  CC_PREFIX="${CC_PREFIX}" NDK_ROOT="${NDK_ROOT}" \
tensorflow/contrib/makefile/compile_android_protobuf.sh -c -a ${ARCH}
fi

# Compile nsync for the host and the target Android device architecture.
# Don't use  export var=`something` syntax; it swallows the exit status.
HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
TARGET_NSYNC_LIB=`CC_PREFIX="${CC_PREFIX}" NDK_ROOT="${NDK_ROOT}" \
      tensorflow/contrib/makefile/compile_nsync.sh -t android -a ${ARCH}`
export HOST_NSYNC_LIB TARGET_NSYNC_LIB

if [[ ! -z "${HEXAGON_LIB_PATH}" ]]; then
    echo "Copy hexagon libraries from ${HEXAGON_LIB_PATH}"

    mkdir -p "${HEXAGON_DOWNLOAD_PATH}/libs"
    cp -fv "${HEXAGON_LIB_PATH}/libhexagon_controller.so" \
"${HEXAGON_DOWNLOAD_PATH}/libs/libhexagon_controller.so"
    cp -fv "${HEXAGON_LIB_PATH}/libhexagon_nn_skel.so" \
"${HEXAGON_DOWNLOAD_PATH}/libs/libhexagon_nn_skel.so"

    USE_HEXAGON="true"
fi

if [[ "${USE_HEXAGON}" == "true" ]]; then
    HEXAGON_PARENT_DIR=$(cd "${HEXAGON_DOWNLOAD_PATH}" >/dev/null && pwd)
    HEXAGON_LIBS="${HEXAGON_PARENT_DIR}/libs"
    HEXAGON_INCLUDE=$(cd "tensorflow/core/kernels/hexagon" >/dev/null && pwd)
fi

if [[ "${ENABLE_EXPERIMENTAL_HEXNN_OPS}" == "true" ]]; then
    EXTRA_MAKE_ARGS+=("ENABLE_EXPERIMENTAL_HEXNN_OPS=true")
fi

if [[ -z "${BUILD_TARGET}" ]]; then
    make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
         TARGET=ANDROID NDK_ROOT="${NDK_ROOT}" ANDROID_ARCH="${ARCH}" \
         CC_PREFIX="${CC_PREFIX}" \
         HOST_NSYNC_LIB="$HOST_NSYNC_LIB" TARGET_NSYNC_LIB="$TARGET_NSYNC_LIB" \
HEXAGON_LIBS="${HEXAGON_LIBS}" HEXAGON_INCLUDE="${HEXAGON_INCLUDE}" \
SUB_MAKEFILES="${SUB_MAKEFILES}" ${EXTRA_MAKE_ARGS[@]}
else
    # BUILD_TARGET explicitly uncommented to allow multiple targets to be
    # passed to make in a single build_all_android.sh invocation.
    make -j"${JOB_COUNT}" -f tensorflow/contrib/makefile/Makefile \
         TARGET=ANDROID NDK_ROOT="${NDK_ROOT}" ANDROID_ARCH="${ARCH}" \
         CC_PREFIX="${CC_PREFIX}" \
         HOST_NSYNC_LIB="$HOST_NSYNC_LIB" TARGET_NSYNC_LIB="$TARGET_NSYNC_LIB" \
HEXAGON_LIBS="${HEXAGON_LIBS}" HEXAGON_INCLUDE="${HEXAGON_INCLUDE}" \
SUB_MAKEFILES="${SUB_MAKEFILES}" ${EXTRA_MAKE_ARGS[@]} ${BUILD_TARGET}
fi
