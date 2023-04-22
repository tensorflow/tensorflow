#!/bin/bash
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

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

# From this point on, logs can be publicly available
set -x

source tensorflow/tools/ci_build/release/common.sh
install_bazelisk
which bazel

tag_filters="gpu,-no_gpu,-nogpu,-benchmark-test,-no_oss,-oss_serial,-no_gpu_presubmit,-no_cuda11""$(maybe_skip_v1)"

# Get the default test targets for bazel.
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Run bazel test command.
"${BAZEL_WRAPPER_PATH}" \
  test \
  --config=rbe_linux_cuda_nvcc_py36 \
  --config=tensorflow_testing_rbe_linux \
  --test_tag_filters="${tag_filters}" \
  --build_tag_filters="${tag_filters}" \
  --test_lang_filters=cc,py \
  -- \
  ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/...

# Copy log to output to be available to GitHub
ls -la "$(bazel info output_base)/java.log"
cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"

