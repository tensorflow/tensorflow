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

if [[ -z "${NDK_ROOT}" ]]; then
    echo "NDK_ROOT should be set as an environment variable" 1>&2
    exit 1
fi

target_host=""
toolchain_path=""

usage() {
    echo "Usage: $(basename "$0") [t:h:c]"
    echo "-t Absolute path to a toolchain"
    echo "-h Target host"
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

if [[ -z "${toolchain_path}" ]]
then
echo "You need to specify toolchain path. Use -t"
exit 1
fi

if [[ -z "${target_host}" ]]
then
echo "You need to specify target host. Use -h"
exit 1
fi

# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${SCRIPT_DIR}"/../../../

source "${SCRIPT_DIR}/build_helper.subr"
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

HEXAGON_DOWNLOAD_PATH="tensorflow/contrib/makefile/downloads/hexagon"

# Remove any old files first.
make -f tensorflow/contrib/makefile/Makefile cleantarget
rm -rf tensorflow/contrib/makefile/downloads
tensorflow/contrib/makefile/download_dependencies.sh
tensorflow/contrib/makefile/compile_android_protobuf.sh -t ${toolchain_path} -h ${target_host}
HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_android_nsync.sh -t ${toolchain_path}`
export HOST_NSYNC_LIB TARGET_NSYNC_LIB

make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID ANDROID_TOOLCHAIN=${toolchain_path} ANDROID_HOST={target_host}
