#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# Creates the project file distributions for the TensorFlow Lite Micro test and
# example targets aimed at embedded platforms.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../../..
cd ${ROOT_DIR}
pwd

make -f tensorflow/lite/experimental/micro/tools/make/Makefile \
  clean clean_downloads

make -f tensorflow/lite/experimental/micro/tools/make/Makefile \
  TARGET=mbed \
  TAGS="portable_optimized disco_f746ng" \
  generate_projects

tensorflow/lite/experimental/micro/tools/ci_build/install_mbed_cli.sh

for PROJECT_PATH in tensorflow/lite/experimental/micro/tools/make/gen/mbed_*/prj/*/mbed; do
  PROJECT_PARENT_DIR=$(dirname ${PROJECT_PATH})
  PROJECT_NAME=$(basename ${PROJECT_PARENT_DIR})
  # Don't try to build and package up test projects, because there are too many.
  if [[ ${PROJECT_NAME} == *"_test" ]]; then
    continue
  fi
  cp -r ${PROJECT_PATH} ${PROJECT_PARENT_DIR}/${PROJECT_NAME}
  pushd ${PROJECT_PARENT_DIR}
  zip -q -r ${PROJECT_NAME}.zip ${PROJECT_NAME}
  popd
  tensorflow/lite/experimental/micro/tools/ci_build/test_mbed_library.sh ${PROJECT_PATH}
done

# Needed to solve CI build bug triggered by files added to source tree.
make -f tensorflow/lite/experimental/micro/tools/make/Makefile clean_downloads
