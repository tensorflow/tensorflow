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

install_ubuntu_16_python_pip_deps python3.9
pip3.9 install --user --upgrade --ignore-installed cloud-tpu-client
install_bazelisk

test_patterns=(//tensorflow/... -//tensorflow/compiler/... -//tensorflow/lite/...)
tag_filters="tpu,-tpu_pod,-no_tpu,-notpu,-no_oss,-no_oss_py37"

bazel_args=(
  --config=release_cpu_linux \
  --repo_env=PYTHON_BIN_PATH="$(which python3.9)" \
  --build_tag_filters="${tag_filters}" \
  --test_sharding_strategy=disabled \
  --test_tag_filters="${tag_filters}" \
  --test_output=errors --verbose_failures=true --keep_going \
  --build_tests_only
)

bazel build "${bazel_args[@]}" -- "${test_patterns[@]}"

TPU_NAME="kokoro-tpu-2vm-${RANDOM}"
TPU_PROJECT="tensorflow-testing-tpu"
TPU_ZONES="us-central1-b:v2-8 us-central1-c:v2-8 us-central1-b:v3-8 us-central1-a:v3-8"

for TPU_ZONE_WITH_TYPE in $TPU_ZONES; do
  TPU_ZONE="$(echo "${TPU_ZONE_WITH_TYPE}" | cut -d : -f 1)"
  TPU_TYPE="$(echo "${TPU_ZONE_WITH_TYPE}" | cut -d : -f 2)"
  if gcloud compute tpus create "$TPU_NAME" \
    --zone="${TPU_ZONE}" \
    --accelerator-type="${TPU_TYPE}" \
    --version=nightly; then
    TPU_CREATED="true"
    break
  fi
done

if [[ ! "${TPU_CREATED}" == "true" ]]; then
  exit 1
fi

# Clean up script uses these files.
echo "${TPU_NAME}" > "${KOKORO_ARTIFACTS_DIR}/tpu_name"
echo "${TPU_ZONE}" > "${KOKORO_ARTIFACTS_DIR}/tpu_zone"
echo "${TPU_PROJECT}" > "${KOKORO_ARTIFACTS_DIR}/tpu_project"

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
