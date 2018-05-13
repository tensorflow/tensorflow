#!/bin/bash -e
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
# Builds protobuf 3 for Android. Pass in the location of your NDK as the first
# argument to the script, for example:
# tensorflow/contrib/makefile/compile_android_protobuf.sh \
# ${HOME}/toolchains/clang-21-stl-gnu

target_host=aarch64-linux-android
toolchain_path=""

usage() {
  echo "Usage: $(basename "$0") [t:h:c]"
  echo "-t Absolute path to a toolchainx"
  echo "-h Target host. Default value is aarch64-linux-android"
  echo "-c Clean before building protobuf for target"
  echo "\"NDK_ROOT\" should be defined as an environment variable."
  exit 1
}

SCRIPT_DIR=$(dirname $0)

# debug options
while getopts "h:t:c" opt_name; do
  case "$opt_name" in
    t) toolchain_path="${OPTARG}";;
    h) target_host="${OPTARG}";;
    c) clean=true;;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

source "${SCRIPT_DIR}/build_helper.subr"
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

if [[ -z "${toolchain_path}" ]]
then
  echo "You need to specify toolchain path. Use -t"
  exit 1
fi

if [[ -z "${NDK_ROOT}" ]]
then
  echo "You need to pass in the Android NDK location as the environment \
variable"
  echo "e.g. NDK_ROOT=${HOME}/android_ndk/android-ndk-rXXx \
tensorflow/contrib/makefile/compile_android_protobuf.sh"
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/Makefile" ]]; then
    echo "Makefile not found in ${SCRIPT_DIR}" 1>&2
    exit 1
fi

cd "${SCRIPT_DIR}"
if [ $? -ne 0 ]
then
    echo "cd to ${SCRIPT_DIR} failed." 1>&2
    exit 1
fi

GENDIR="$(pwd)/gen/protobuf_android"
HOST_GENDIR="$(pwd)/gen/protobuf-host"
DIST_DIR="${GENDIR}/distribution"
mkdir -p "${GENDIR}"
mkdir -p "${DIST_DIR}"

if [[ ! -f "./downloads/protobuf/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "tensorflow/contrib/makefile/download_dependencies.sh" 1>&2
    exit 1
fi

cd downloads/protobuf

PROTOC_PATH="${HOST_GENDIR}/bin/protoc"
if [[ ! -f "${PROTOC_PATH}" || ${clean} == true ]]; then
  # Try building compatible protoc first on host
  echo "protoc not found at ${PROTOC_PATH}. Build it first."
  make_host_protoc "${HOST_GENDIR}"
else
  echo "protoc found. Skip building host tools."
fi

export SYSROOT="${toolchain_path}/sysroot"
export CC="${toolchain_path}/bin/clang --sysroot ${SYSROOT}"
export CXX="${toolchain_path}/bin/clang++ --sysroot ${SYSROOT}"

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

./configure --prefix="${DIST_DIR}" \
--host="${target_host}" \
--with-sysroot="${SYSROOT}" \
--disable-shared \
--enable-cross-compile \
--with-protoc="${PROTOC_PATH}" \
CFLAGS="${march_option}" \
LIBS="-llog -lz -lc++_static"

if [ $? -ne 0 ]
then
  echo "./configure command failed."
  exit 1
fi

if [[ ${clean} == true ]]; then
  echo "clean before build"
  make clean
fi

make -j"${JOB_COUNT}"
if [ $? -ne 0 ]
then
  echo "make command failed."
  exit 1
fi

make install

echo "$(basename $0) finished successfully!!!"
