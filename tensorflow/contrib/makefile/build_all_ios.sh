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
  echo "-g [graph] optimize and selectively register ops only for this graph"
  echo "-T only build tensorflow (dont download other deps etc)"
  exit 1
}

echo "********************************************************************"
echo "TensorFlow Lite is the recommended library for mobile and embedded machine learning inference."
echo "You are currently using an older version. Please switch over to TensorFlow Lite."
echo ""
echo "Link to the code: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite"
echo "********************************************************************"
echo ""

DEFAULT_ARCH="i386 x86_64 armv7 armv7s arm64"
while getopts "a:g:T" opt_name; do
  case "$opt_name" in
    a) BUILD_ARCH="${OPTARG}";;
    g) OPTIMIZE_FOR_GRAPH="${OPTARG}";;
    T) ONLY_MAKE_TENSORFLOW="true";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))


# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_SRCDIR="${SCRIPT_DIR}/../../../"
cd ${TOP_SRCDIR}

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

PRNT_SLCTV_BIN="${TOP_SRCDIR}bazel-bin/tensorflow/python/tools/print_selective_registration_header"

if [[ ! -z "${OPTIMIZE_FOR_GRAPH}" ]]; then
    echo "Request to optimize for graph: ${OPTIMIZE_FOR_GRAPH}"
    #Request to trim the OPs by selectively registering
    if [ ! -f ${PRNT_SLCTV_BIN} ]; then
        #Build bazel build tensorflow/python/tools:print_selective_registration_header
        echo "${PRNT_SLCTV_BIN} not found. Trying to build it"
        cd ${TOP_SRCDIR}
        bazel build --copt="-DUSE_GEMM_FOR_CONV" tensorflow/python/tools:print_selective_registration_header
         if [ ! -f ${PRNT_SLCTV_BIN} ]; then
            echo "Building print_selective_registration_header failed"
            echo "You may want to build TensorFlow with: "
            echo "./configure"
            echo "bazel build --copt="-DUSE_GEMM_FOR_CONV" tensorflow/python/tools:print_selective_registration_header"
            echo "and then run this script again"
            exit 1
        fi
    else
        echo "${PRNT_SLCTV_BIN} found. Using it"
    fi

    ${PRNT_SLCTV_BIN} --graphs=${OPTIMIZE_FOR_GRAPH} > ${TOP_SRCDIR}/tensorflow/core/framework/ops_to_register.h
fi

if [[ "${ONLY_MAKE_TENSORFLOW}" != "true" ]]; then
    # Remove any old files first.
    make -f tensorflow/contrib/makefile/Makefile clean
    rm -rf tensorflow/contrib/makefile/downloads

    # Pull down the required versions of the frameworks we need.
    tensorflow/contrib/makefile/download_dependencies.sh

    if [[ -z "${BUILD_ARCH}" ]]; then
        # Compile protobuf for the target iOS device architectures.
        tensorflow/contrib/makefile/compile_ios_protobuf.sh
    else
        # Compile protobuf for the target iOS device architectures.
        tensorflow/contrib/makefile/compile_ios_protobuf.sh -a ${BUILD_ARCH}
    fi
fi

# Compile nsync for the target iOS device architectures.
# Don't use  export var=`something` syntax; it swallows the exit status.
HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
if [[ -z "${BUILD_ARCH}" ]]; then
    # No arch specified so build all architectures
    TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh -t ios`
else
    # arch specified so build just that
    TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh -t ios -a "${BUILD_ARCH}"`
fi
export HOST_NSYNC_LIB TARGET_NSYNC_LIB

TF_CC_FLAGS="-O3"
TF_SCRIPT_FLAGS="-h ${HOST_NSYNC_LIB} -n ${TARGET_NSYNC_LIB}"

if [[ ! -z "${OPTIMIZE_FOR_GRAPH}" ]]; then
    # arch specified so build just that
    TF_CC_FLAGS="${TF_CC_FLAGS} -DANDROID_TYPES=__ANDROID_TYPES_FULL__ -DSELECTIVE_REGISTRATION -DSUPPORT_SELECTIVE_REGISTRATION"
    # The Makefile checks the env var to decide which ANDROID_TYPES to build
    export ANDROID_TYPES="-D__ANDROID_TYPES_FULL__"
fi

if [[ ! -z "${BUILD_ARCH}" ]]; then
    # arch specified so build just that
    TF_SCRIPT_FLAGS="${TF_SCRIPT_FLAGS} -a ${BUILD_ARCH}"
fi

# build the ios tensorflow libraries.
echo "Building TensorFlow with flags: ${TF_SCRIPT_FLAGS} -f ${TF_CC_FLAGS}"
tensorflow/contrib/makefile/compile_ios_tensorflow.sh ${TF_SCRIPT_FLAGS} -f "${TF_CC_FLAGS}"

# Creates a static universal library in
# tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a
