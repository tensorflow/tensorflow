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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# explicitly call third_party_downloads since we need pigweed for the license
# and clang-format checks.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Explicitly disable exit on error so that we can properly clean up the
# temporary git repository even when one of the scripts fail with an error code.
set +e

# The pigweed scripts only work from a git repository and the Tensorflow CI
# infrastructure does not always guarantee that. As an ugly workaround, we
# create our own git repo when running on the CI servers.
pushd tensorflow/lite/micro/
if [[ ${1} == "PRESUBMIT" ]]; then
  git init .
  git config user.email "tflm@google.com"
  git config user.name "TensorflowLite Micro"
  git add *
  git commit -a -m "Commit for a temporary repository." > /dev/null
fi

# Check for license with the necessary exclusions.
tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/pigweed_presubmit.py \
  . \
  -p copyright_notice \
  -e tools/make/downloads \
  -e tools/make/targets/ecm3531 \
  -e BUILD\
  -e leon_commands \
  -e "\.bzl" \
  -e "\.h5" \
  -e "\.ipynb" \
  -e "\.inc" \
  -e "\.patch" \
  -e "\.properties" \
  -e "\.txt" \
  -e "\.tpl" \
  --output-directory /tmp

LICENSE_CHECK_RESULT=$?

# Check that the TFLM-only code is clang-formatted We are currently ignoring
# Python files (with yapf as the formatter) because that needs additional setup.
# We are also ignoring the markdown files to allow for a more gradual rollout of
# this presubmit check.
tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/format_code.py \
  . \
  -e "\.inc" \
  -e "\.md" \
  -e "\.py"

CLANG_FORMAT_RESULT=$?

popd
if [[ ${1} == "PRESUBMIT" ]]; then
  rm -rf tensorflow/lite/micro/.git
fi

# Re-enable exit on error now that we are done with the temporary git repo.
set -e

if [[ ${LICENSE_CHECK_RESULT} != 0 || ${CLANG_FORMAT_RESULT} != 0 ]]
then
  exit 1
fi
