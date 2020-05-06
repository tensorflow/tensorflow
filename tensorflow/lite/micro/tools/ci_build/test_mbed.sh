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
# This script takes a single argument to differentiate between running it as
# part of presubmit checks or not.
#
# This will generate a subset of targets:
#   test_mbed.sh PRESUBMIT
#
# This will run generate all the targets:
#   test_mbed.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

TARGET=mbed

# We limit the number of projects that we build as part of the presubmit checks
# to keep the overall time low, but build everything as part of the nightly
# builds.
if [[ ${1} == "PRESUBMIT" ]]; then
  PROJECTS="generate_hello_world_mbed_project generate_micro_speech_mbed_project"
else
  PROJECTS=generate_projects
fi

make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TAGS="portable_optimized disco_f746ng" \
  ${PROJECTS}

readable_run tensorflow/lite/micro/tools/ci_build/install_mbed_cli.sh

for PROJECT_PATH in tensorflow/lite/micro/tools/make/gen/mbed_*/prj/*/mbed; do
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
  readable_run tensorflow/lite/micro/tools/ci_build/test_mbed_library.sh ${PROJECT_PATH}
done
