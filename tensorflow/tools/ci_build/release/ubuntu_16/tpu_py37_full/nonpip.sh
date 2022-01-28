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
set -x

source tensorflow/tools/ci_build/release/common.sh
source tensorflow/tools/ci_build/ctpu/ctpu.sh

install_ubuntu_16_python_pip_deps python3.9
install_bazelisk
install_ctpu pip3.9

test_patterns=(//tensorflow/... -//tensorflow/compiler/... -//tensorflow/lite/...)
tag_filters="tpu,-tpu_pod,-no_tpu,-notpu,-no_oss,-no_oss_py37"

bazel_args=(
  --config=release_cpu_linux \
  --repo_env=PYTHON_BIN_PATH="$(which python3.9)" \
  --build_tag_filters="${tag_filters}" \
  --test_sharding_strategy=disabled \
  --test_tag_filters="${tag_filters}" \
  --test_output=errors --verbose_failures=true --keep_going
)

bazel build "${bazel_args[@]}" -- "${test_patterns[@]}"

ctpu_up -s v2-8 -p tensorflow-testing-tpu

test_args=(
  --test_timeout=120,600,-1,-1 \
  --test_arg=--tpu="${TPU_NAME}" \
  --test_arg=--zone="${TPU_ZONE}" \
  --test_arg=--test_dir_base=gs://kokoro-tpu-testing/tempdir/ \
  --local_test_jobs=1
)

set +e
bazel test "${bazel_args[@]}" "${test_args[@]}" -- "${test_patterns[@]}"
test_xml_summary_exit
