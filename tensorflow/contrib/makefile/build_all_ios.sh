#!/usr/bin/env bash
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

set -e

# Make sure we're on OS X.
if [[ $(uname) != "Darwin" ]]; then
    echo "ERROR: This makefile build requires macOS, which the current system "\
    "is not."
    exit 1
fi

usage() {
  echo "Usage: $(basename "$0") [-a:T]"
  echo "-a [build_arch] build only for specified arch x86_64 [default=all]"
  echo "-T only build tensorflow (dont download other deps etc)"
  exit 1
}

while getopts "a:T" opt_name; do
  case "$opt_name" in
    a) BUILD_ARCH="${OPTARG}";;
    T) ONLY_MAKE_TENSORFLOW="true";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))


# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}/../../../

source "${SCRIPT_DIR}/build_helper.subr"
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

# Setting a deployment target is required for building with bitcode,
# otherwise linking will fail with:
#
#    ld: -bind_at_load and -bitcode_bundle (Xcode setting ENABLE_BITCODE=YES) cannot be used together
#
if [[ -n MACOSX_DEPLOYMENT_TARGET ]]; then
    export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
fi

if [[ "${ONLY_MAKE_TENSORFLOW}" != "true" ]]; then
    # Remove any old files first.
    make -f tensorflow/contrib/makefile/Makefile clean
    rm -rf tensorflow/contrib/makefile/downloads

    # Pull down the required versions of the frameworks we need.
    tensorflow/contrib/makefile/download_dependencies.sh

    # Compile protobuf for the target iOS device architectures.
    tensorflow/contrib/makefile/compile_ios_protobuf.sh
fi

# Compile nsync for the target iOS device architectures.
# Don't use  export var=`something` syntax; it swallows the exit status.
HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
if [[ -z "${BUILD_ARCH}" ]]; then
    # No arch specified so build all architectures
    TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh -t ios`
else
    # arch specified so build just that
    TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh -t ios -a ${BUILD_ARCH}`
fi
export HOST_NSYNC_LIB TARGET_NSYNC_LIB

if [[ -z "${BUILD_ARCH}" ]]; then
    # build the ios tensorflow libraries.
    tensorflow/contrib/makefile/compile_ios_tensorflow.sh -f "-O3" -h $HOST_NSYNC_LIB -n $TARGET_NSYNC_LIB
else
    # arch specified so build just that
    tensorflow/contrib/makefile/compile_ios_tensorflow.sh -f "-O3" -a "${BUILD_ARCH}" -h $HOST_NSYNC_LIB -n $TARGET_NSYNC_LIB
fi

# Creates a static universal library in
# tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a
