#!/usr/bin/env bash
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
# This is a composite script to build and run inception with hexagon on Android

set -e

usage() {
    echo "Usage: QUALCOMM_SDK=<path to qualcomm sdk> NDK_ROOT=<path to ndk root> $(basename "$0")"
    exit 1
}

if [[ -z "${NDK_ROOT}" ]]; then
    echo "NDK_ROOT is empty" 1>&2
    usage
    exit 1
fi

if [[ -z "${QUALCOMM_SDK}" ]]; then
    echo "QUALCOMM_SDK is empty" 1>&2
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_ALL_ANDROID_PATH="${SCRIPT_DIR}/../build_all_android.sh"
GEN_DIR="${SCRIPT_DIR}/gen"
GEN_LIBS_DIR="${GEN_DIR}/libs"
GEN_DOWNLOAD_DIR="${GEN_DIR}/downloads"

source "${SCRIPT_DIR}/../build_helper.subr"

rm -rf "${GEN_DIR}"
mkdir -p "${GEN_LIBS_DIR}"
mkdir -p "${GEN_DOWNLOAD_DIR}"

cd "${GEN_DIR}"
git clone https://source.codeaurora.org/quic/hexagon_nn/nnlib

cd "${QUALCOMM_SDK}"
source "${QUALCOMM_SDK}/setup_sdk_env.sh"

GENERATED_NNLIB_DIRECTORY="${QUALCOMM_SDK}/examples/common/generated_nnlib"
if [[ -d "${GENERATED_NNLIB_DIRECTORY}" ]]; then
    echo "Existing nnlib found.  Remove"
    rm -rf "${GENERATED_NNLIB_DIRECTORY}"
fi

cp -af "${GEN_DIR}/nnlib" "${GENERATED_NNLIB_DIRECTORY}"
cd "${GENERATED_NNLIB_DIRECTORY}"

make clean V=hexagon_Release_dynamic_toolv72_v60
rm -rf hexagon_Release_dynamic_toolv72_v60
make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv72_v60

GENERATED_HEXAGON_CONTROLLER_DIRECTORY=\
"${QUALCOMM_SDK}/examples/common/generated_hexagon_controller"

if [[ -d "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}" ]]; then
    echo "Existing hexagon controller found.  Remove"
    rm -rf "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}"
fi

cp -af "${TF_ROOT_DIR}/tensorflow/contrib/hvx/hexagon_controller" \
   "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}"

echo "Copy interface directory"
cp -afv "${GENERATED_NNLIB_DIRECTORY}/interface" \
"${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/"

echo "Copy glue directory"
cp -afv "${GENERATED_NNLIB_DIRECTORY}/glue" \
"${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/"

cd "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}"
make clean V=android_Release
rm -rf android_Release
make tree VERBOSE=1 V=android_Release

cp -v "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/android_Release/ship/libhexagon_controller.so" "${GEN_LIBS_DIR}"
cp -v "${GENERATED_NNLIB_DIRECTORY}/hexagon_Release_dynamic_toolv72_v60/ship/libhexagon_nn_skel.so" "${GEN_LIBS_DIR}"

if [[ -d "${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf" &&
      -d "${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf-host" ]]; then
    echo "generated protobuf and protobuf-host found."
    extra_args+=("-T")
fi

if [[ -z "${CC_PREFIX}" ]]; then
    echo "HINT: Installing ccache and specifying CC_PREFIX=ccache accelerate build time"
fi

CC_PREFIX=${CC_PREFIX} NDK_ROOT=${NDK_ROOT} "${BUILD_ALL_ANDROID_PATH}" \
-x "${GEN_LIBS_DIR}" \
-s "${TF_ROOT_DIR}/tensorflow/contrib/makefile/sub_makefiles/hexagon_graph_execution/Makefile.in" \
-t hexagon_graph_execution ${extra_args[@]}

echo "Download and push inception image"
URL_BASE="https://storage.googleapis.com/download.tensorflow.org"
HEXAGON_DOWNLOAD_PATH=\
"${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads/hexagon"
rm -rf "${HEXAGON_DOWNLOAD_PATH}"
mkdir -p "${HEXAGON_DOWNLOAD_PATH}/libs"

download_and_push "${URL_BASE}/example_images/img_299x299.bmp" \
"${GEN_DOWNLOAD_DIR}/img_299x299.bmp" "/data/local/tmp"

download_and_push \
"${URL_BASE}/models/tensorflow_inception_v3_stripped_optimized_quantized.pb" \
"${GEN_DOWNLOAD_DIR}/tensorflow_inception_v3_stripped_optimized_quantized.pb" \
"/data/local/tmp"

download_and_push "${URL_BASE}/models/imagenet_comp_graph_label_strings.txt" \
"${GEN_DOWNLOAD_DIR}/imagenet_comp_graph_label_strings.txt" "/data/local/tmp"

# By default this script runs a test to fuse and run the model
gtest_args+=("--gtest_filter=GraphTransferer.RunInceptionV3OnHexagonExampleWithTfRuntime")
# Uncomment this block if you want to run the fused model
#gtest_args+=("--gtest_filter=GraphTransferer.RunInceptionV3OnHexagonExampleWithFusedGraph")
# Uncomment this block if you want to run the model with hexagon wrapper
#gtest_args+=(
#    "--gtest_also_run_disabled_tests"
#    "--gtest_filter=GraphTransferer.DISABLED_RunInceptionV3OnHexagonExampleWithHexagonWrapper")
# Uncomment this block if you want to get the list of tests
#gtest_args+=("--gtest_list_tests")

ANDROID_EXEC_FILE_MODE=755

adb push "${GEN_LIBS_DIR}/libhexagon_controller.so" "/data/local/tmp"
adb push "${GEN_LIBS_DIR}/libhexagon_nn_skel.so" "/vendor/lib/rfsa/adsp"

echo "Run hexagon_graph_execution"
adb push -p \
"${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/bin/hexagon_graph_execution" \
"/data/local/tmp/"
adb wait-for-device
adb shell chmod "${ANDROID_EXEC_FILE_MODE}" \
"/data/local/tmp/hexagon_graph_execution"
adb wait-for-device
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH' \
"/data/local/tmp/hexagon_graph_execution" ${gtest_args[@]}
