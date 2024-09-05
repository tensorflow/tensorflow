#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    --input_models=model1.tflite,model2.tflite \\"
  echo "    --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \\"
  echo "    --src_dir=$PWD \\"
  echo "    [--cache_dir=<path to cache directory>]"
  echo ""
  echo "Where: "
  echo "  --input_models: Supported TFLite models. "
  echo "  --target_archs: Supported arches included in the aar file."
  echo "  --src_dir: Directory of tensorflow source code."
  echo "  --cache_dir: Path to the directory to store bazel cache. If not "
  echo "        provided, a directory name bazel-build-cache will be created."
  echo ""
  exit 1
}

# Check command line flags.
ARGUMENTS=$@
BUILD_FLAGS=""
TARGET_ARCHS=x86,x86_64,arm64-v8a,armeabi-v7a
FLAG_SRC_DIR=""

if [ "$#" -gt 5 ]; then
  echo "ERROR: Too many arguments."
  print_usage
fi

for i in "$@"
do
case $i in
    --input_models=*)
      FLAG_MODELS="${i#*=}"
      BUILD_FLAGS="${BUILD_FLAGS} ${i}"
      shift;;
    --target_archs=*)
      TARGET_ARCHS="${i#*=}"
      BUILD_FLAGS="${BUILD_FLAGS} ${i}"
      shift;;
    --src_dir=*)
      FLAG_SRC_DIR="${i#*=}"
      shift;;
    --cache_dir=*)
      BAZEL_CACHE_DIR="${i#*=}"
      shift;;
    --debug)
      DEBUG_MODE=true
      shift;;
    *)
      echo "ERROR: Unrecognized argument: ${i}"
      print_usage;;
esac
done

if [ ! -d /tensorflow_src ]; then
  # Running on host.
  for model in $(echo ${FLAG_MODELS} | sed "s/,/ /g")
  do
    FLAG_DIR="${FLAG_DIR} -v ${model}:${model}"
  done

  if [ -z ${BAZEL_CACHE_DIR} ]; then
    mkdir -p "bazel-build-cache"
    BAZEL_CACHE_DIR="$PWD/bazel-build-cache"
    ARGUMENTS="${ARGUMENTS} --cache_dir=${BAZEL_CACHE_DIR}"
  fi
  FLAG_DIR="${FLAG_DIR} -v ${BAZEL_CACHE_DIR}:${BAZEL_CACHE_DIR}"

  docker run --rm -it -v ${FLAG_SRC_DIR}:/tensorflow_src -v $PWD:/host_dir \
    -v ${SCRIPT_DIR}:/script_dir ${FLAG_DIR} \
    --entrypoint /script_dir/build_aar_with_docker.sh tflite-builder \
    ${ARGUMENTS}
  exit 0
else
  # Running inside docker container, download the SDK first.
  sdkmanager --licenses
  sdkmanager \
    "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
    "platform-tools" \
    "platforms;android-${ANDROID_API_LEVEL}"

  cd /tensorflow_src

  # Run configure.
  # -Wno-c++20-designator can be removed once tf supports C++20.
  # -Wno-gnu-inline-cpp-without-extern is needed for NEON2SSE. Can remove after
  # https://github.com/intel/ARM_NEON_2_x86_SSE/issues/57 is resolved.
  configs=(
    '/usr/bin/python3'
    '/usr/lib/python3/dist-packages'
    'N'
    'N'
    'Y'
    '/usr/lib/llvm-18/bin/clang'
    '-Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'
    'y'
    '/android/sdk'
  )
  printf '%s\n' "${configs[@]}" | ./configure

  # Configure Bazel.
  source tensorflow/tools/ci_build/release/common.sh
  install_bazelisk

  # Building with bazel.
  export BAZEL_CACHE_DIR=${BAZEL_CACHE_DIR}
  export OMIT_PRINTING_OUTPUT_PATHS=YES
  if [ "${DEBUG_MODE}" = true ]; then
    echo "### Run /tensorflow_src/tensorflow/lite/tools/build_aar.sh ${BUILD_FLAGS}"
    bash -i
    exit 0
  else
    bash /tensorflow_src/tensorflow/lite/tools/build_aar.sh ${BUILD_FLAGS}
  fi

  # Copy the output files from docker container.
  OUT_FILES="/tensorflow_src/bazel-bin/tmp/tensorflow-lite.aar"
  OUT_FILES="${OUT_FILES} /tensorflow_src/bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar"
  echo "Output can be found here:"
  for i in ${OUT_FILES}
  do
    if [ -f $i ]; then
      cp $i /host_dir
      basename $i
    fi
  done
fi
