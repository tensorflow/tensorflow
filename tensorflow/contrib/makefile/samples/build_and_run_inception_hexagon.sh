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
  echo "Usage: QUALCOMM_SDK=<path to qualcomm sdk. Not needed if you specify -p> NDK_ROOT=<path to ndk root> $(basename "$0")"
  echo "Optional: NNLIB_DIR=<path to downloaded nnlib dir>"
  echo "-b build only"
  echo "-c test count"
  echo "-E enable experimental hexnn ops"
  echo "-p use prebuilt hexagon binaries"
  echo "-s skip download if files already exist"
  exit 1
}

TEST_COUNT=1
SKIP_DOWNLOAD_IF_EXIST=false

while getopts "bc:Eps" opt_name; do
  case "$opt_name" in
    b) BUILD_ONLY="true";;
    c) TEST_COUNT="${OPTARG}";;
    E) ENABLE_EXPERIMENTAL_HEXNN_OPS="true";;
    p) USE_PREBUILT_HEXAOGON_BINARIES="true";;
    s) SKIP_DOWNLOAD_IF_EXIST="true";;
    *) usage;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "${NDK_ROOT}" ]]; then
    echo "NDK_ROOT is empty" 1>&2
    usage
    exit 1
fi

if [[ "${USE_PREBUILT_HEXAOGON_BINARIES}" != "true" &&
      -z "${QUALCOMM_SDK}" ]]; then
    echo "QUALCOMM_SDK is empty" 1>&2
    usage
    exit 1
fi

if [[ "${BUILD_ONLY}" != "true" ]]; then
    if ! type adb >/dev/null 2>&1; then
        echo "adb is not in your path ${PATH}."
        exit 1
    fi
    if ! adb shell ls /system/lib/rfsa/adsp/testsig* >/dev/null 2>&1; then
        echo "test signature not found. Unlock your phone first"
        echo "See ${QUALCOMM_SDK}/tools/elfsigner/README.txt"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
TF_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_ALL_ANDROID_PATH="${SCRIPT_DIR}/../build_all_android.sh"
GEN_DIR="${SCRIPT_DIR}/gen"
GEN_LIBS_DIR="${GEN_DIR}/libs"
GEN_DOWNLOAD_DIR="${GEN_DIR}/downloads"
URL_BASE="https://storage.googleapis.com/download.tensorflow.org"

ARCH="armeabi-v7a"

source "${SCRIPT_DIR}/../build_helper.subr"

rm -rf "${GEN_DIR}"
mkdir -p "${GEN_LIBS_DIR}"
mkdir -p "${GEN_DOWNLOAD_DIR}"

if [[ "${USE_PREBUILT_HEXAOGON_BINARIES}" == "true" ]]; then
    echo "Download prebuilt hexagon binaries"
    if [[ "${BUILD_ONLY}" != "true" ]]; then
        CONTROLLER_PUSH_DEST="/data/local/tmp"
        NN_LIB_PUSH_DEST="/vendor/lib/rfsa/adsp"
    fi
    download_and_push "${URL_BASE}/deps/hexagon/libhexagon_controller.so" \
"${GEN_LIBS_DIR}/libhexagon_controller.so" "${CONTROLLER_PUSH_DEST}" \
"${SKIP_DOWNLOAD_IF_EXIST}"

    download_and_push "${URL_BASE}/deps/hexagon/libhexagon_nn_skel.so" \
"${GEN_LIBS_DIR}/libhexagon_nn_skel.so" "${NN_LIB_PUSH_DEST}" \
"${SKIP_DOWNLOAD_IF_EXIST}"
else
    echo "Build hexagon binaries from source code"
    cd "${GEN_DIR}"
    if [[ -z "${NNLIB_DIR}" ]]; then
      git clone https://source.codeaurora.org/quic/hexagon_nn/nnlib
    else
      if [[ ! -f "${NNLIB_DIR}/Makefile" ]]; then
        echo "Couldn't locate ${NNLIB_DIR}/Makefile" 1>&2
        exit 1
      fi
      echo "Use nnlib in ${NNLIB_DIR}" 1>&2
      GEN_NNLIB_DIR="${GEN_DIR}/nnlib"
      mkdir -p "${GEN_NNLIB_DIR}"
      cp -af "${NNLIB_DIR}/"* "${GEN_NNLIB_DIR}"
    fi

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

    cp -v "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/android_Release/ship/libhexagon_controller.so" \
       "${GEN_LIBS_DIR}"
    cp -v "${GENERATED_NNLIB_DIRECTORY}/hexagon_Release_dynamic_toolv72_v60/ship/libhexagon_nn_skel.so" \
       "${GEN_LIBS_DIR}"
fi

if [[ -d "${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf" &&
      -d "${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf-host" ]]; then
    echo "generated protobuf and protobuf-host found."
    EXTRA_ARGS+=("-T")
fi

if [[ "${ENABLE_EXPERIMENTAL_HEXNN_OPS}" == "true" ]]; then
    EXTRA_ARGS+=("-E")
fi

if [[ -z "${CC_PREFIX}" ]]; then
    echo "HINT: Installing ccache and specifying CC_PREFIX=ccache accelerate build time"
fi

CC_PREFIX=${CC_PREFIX} NDK_ROOT=${NDK_ROOT} "${BUILD_ALL_ANDROID_PATH}" \
-x "${GEN_LIBS_DIR}" \
-s "${TF_ROOT_DIR}/tensorflow/contrib/makefile/sub_makefiles/hexagon_graph_execution/Makefile.in" \
-t "hexagon_graph_execution" ${EXTRA_ARGS[@]}

echo "Download and push inception image"
HEXAGON_DOWNLOAD_PATH=\
"${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads/hexagon"
rm -rf "${HEXAGON_DOWNLOAD_PATH}"
mkdir -p "${HEXAGON_DOWNLOAD_PATH}/libs"

if [[ "${BUILD_ONLY}" != "true" ]]; then
    BIN_PUSH_DEST="/data/local/tmp"
fi

download_and_push "${URL_BASE}/example_images/img_299x299.bmp" \
"${GEN_DOWNLOAD_DIR}/img_299x299.bmp" "${BIN_PUSH_DEST}" \
"${SKIP_DOWNLOAD_IF_EXIST}"

download_and_push \
"${URL_BASE}/models/tensorflow_inception_v3_stripped_optimized_quantized.pb" \
"${GEN_DOWNLOAD_DIR}/tensorflow_inception_v3_stripped_optimized_quantized.pb" \
"${BIN_PUSH_DEST}" \
"${SKIP_DOWNLOAD_IF_EXIST}"

download_and_push "${URL_BASE}/models/imagenet_comp_graph_label_strings.txt" \
"${GEN_DOWNLOAD_DIR}/imagenet_comp_graph_label_strings.txt" "${BIN_PUSH_DEST}" \
"${SKIP_DOWNLOAD_IF_EXIST}"

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

if [[ "${BUILD_ONLY}" != "true" ]]; then
    echo "Run hexagon_graph_execution"
    ANDROID_EXEC_FILE_MODE=755

    adb push "${GEN_LIBS_DIR}/libhexagon_controller.so" "/data/local/tmp"
    adb push "${GEN_LIBS_DIR}/libhexagon_nn_skel.so" "/vendor/lib/rfsa/adsp"

    adb push -p \
        "${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/bin/android_${ARCH}/hexagon_graph_execution" \
        "/data/local/tmp/"
    adb wait-for-device
    adb shell chmod "${ANDROID_EXEC_FILE_MODE}" \
        "/data/local/tmp/hexagon_graph_execution"
    adb wait-for-device

    for i in $(seq 1 "${TEST_COUNT}"); do
      adb shell 'LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH' \
          "/data/local/tmp/hexagon_graph_execution" ${gtest_args[@]}
    done
fi
