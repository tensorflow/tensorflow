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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    --input_models=model1.tflite,model2.tflite \\"
  echo "    --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \\"
  echo "    --checkpoint=master \\"
  echo "    [--cache_dir=<path to cache directory>]"
  echo ""
  echo "Where: "
  echo "  --input_models: Supported TFLite models. "
  echo "  --target_archs: Supported arches included in the aar file."
  echo "  --checkpoint: Checkpoint of the github repo, could be a branch, a "
  echo "        commit or a tag. Default: lastest release branch."
  echo "  --cache_dir: Path to the directory to store bazel cache. If not "
  echo "        provided, a directory name bazel-build-cache will be created."
  echo ""
  exit 1
}

# Check command line flags.
ARGUMENTS=$@
BUILD_FLAGS=""
TARGET_ARCHS=x86,x86_64,arm64-v8a,armeabi-v7a
FLAG_CHECKPOINT="r2.4" # TODO(b/163918542) Set default to lastest release.

if [ "$#" -gt 4 ]; then
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
    --checkpoint=*)
      FLAG_CHECKPOINT="${i#*=}"
      shift;;
    --cache_dir=*)
      BAZEL_CACHE_DIR="${i#*=}"
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

  docker run --rm -it -v $PWD:/host_dir -v ${SCRIPT_DIR}:/script_dir ${FLAG_DIR} \
    --entrypoint /script_dir/build_aar_with_docker.sh tflite-builder \
    ${ARGUMENTS}
  exit 0
else
  # Running inside docker container, download the SDK first.
  android update sdk --no-ui -a \
    --filter tools,platform-tools,android-${ANDROID_API_LEVEL},build-tools-${ANDROID_BUILD_TOOLS_VERSION}

  cd /tensorflow_src

  # Run configure.
  configs=(
    '/usr/bin/python3'
    '/usr/lib/python3/dist-packages'
    'N'
    'N'
    'N'
    'N'
    '-march=native -Wno-sign-compare'
    'y'
    '/android/sdk'
  )
  printf '%s\n' "${configs[@]}" | ./configure

  # Pull the latest code from tensorflow.
  git pull -a
  git checkout ${FLAG_CHECKPOINT}

  # Configure Bazel.
  source tensorflow/tools/ci_build/release/common.sh
  install_bazelisk

  # Building with bazel.
  export BAZEL_CACHE_DIR=${BAZEL_CACHE_DIR}
  export OMIT_PRINTING_OUTPUT_PATHS=YES
  bash /tensorflow_src/tensorflow/lite/tools/build_aar.sh ${BUILD_FLAGS}

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
