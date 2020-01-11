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
# TODO(mihaimaruseac,hyey,ggadde): Convert to py3

set -e

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

# From this point on, logs can be publicly available
set -x

function setup_pip () {
  install_pip2
  python -m virtualenv tf_build_env --system-site-packages
  source tf_build_env/bin/activate
  install_macos_pip_deps
}

function run_build () {
  # Run configure.
  export TF_NEED_CUDA=0
  export PYTHON_BIN_PATH=$(which python2)
  yes "" | $PYTHON_BIN_PATH configure.py
  tag_filters="-no_oss,-no_oss_py2,-gpu,-tpu,-benchmark-test,-nomac,-no_mac,-v1only"

  # Get the default test targets for bazel.
  source tensorflow/tools/ci_build/build_scripts/PRESUBMIT_BUILD_TARGETS.sh

  "${BAZEL_WRAPPER_PATH}" \
    test \
    --build_tag_filters="${tag_filters}" \
    --test_tag_filters="${tag_filters}" \
    --action_env=PATH \
    --remote_accept_cached=true \
    --spawn_strategy=standalone \
    --remote_local_fallback=false \
    --remote_timeout=600 \
    --strategy=Javac=standalone \
    --strategy=Closure=standalone \
    --genrule_strategy=standalone \
    -- ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/... -//tensorflow/python:contrib_test -//tensorflow/examples/adding_an_op/...

  # Copy log to output to be available to GitHub
  ls -la "$(bazel info output_base)/java.log"
  cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"
}

source tensorflow/tools/ci_build/release/common.sh
update_bazel_macos
which bazel
set_bazel_outdir

setup_pip
run_build
