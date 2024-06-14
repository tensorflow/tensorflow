#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euox pipefail -o history

# Builds + tests jaxlib against CL/PR version of XLA + JAX main.

source "${KOKORO_GFILE_DIR}/utils.sh"

function is_linux_gpu_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/xla/jax/.*gpu.* ]]
}

clone_main_jax() {
  git clone https://github.com/google/jax.git
}

prelude() {
  export JAX_ENABLE_X64=0

  if is_linux_gpu_job ; then
    export JAX_CUDA_VERSION=12
    export JAX_CUDNN_VERSION=9.1
    nvidia-smi
    setup_env_vars_py39
  else
    setup_env_vars_py312
  fi

  cd "${KOKORO_ARTIFACTS_DIR}"

  use_local_or_install_python
  install_packages "$NUMPY_VERSION" "$SCIPY_VERSION"
  clone_main_jax
  # Install bazel
  update_bazel_linux

  cd jax

}

build_and_test_on_rbe_cpu() {
  # Run the tests.
  bazel \
      test \
      --verbose_failures=true \
      --override_repository=xla="${KOKORO_ARTIFACTS_DIR}"/github/xla \
      --config=avx_posix \
      --config=mkl_open_source_only \
      --config="rbe_cpu_linux_py3.12" \
      --config=tensorflow_testing_rbe_linux \
      --test_env=JAX_NUM_GENERATED_CASES=25 \
      --test_output=errors \
      -- //tests:cpu_tests //tests:backend_independent_tests
}

build_and_test_on_rbe_gpu() {
  # Runs non-multiaccelerator tests with one GPU apiece.
  # It appears --run_under needs an absolute path.

  bazel \
    test \
    --verbose_failures=true \
    --override_repository=xla="${KOKORO_ARTIFACTS_DIR}"/github/xla \
    --config=avx_posix \
    --config=mkl_open_source_only \
    --config="rbe_linux_cuda12.3_nvcc_py3.9" \
    --config=tensorflow_testing_rbe_linux \
    --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --test_output=errors \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
    --test_env=JAX_EXCLUDE_TEST_TARGETS="PmapTest.testSizeOverflow" \
    --test_tag_filters=-multiaccelerator \
    -- //tests:gpu_tests //tests:backend_independent_tests
}

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

prelude

if is_linux_gpu_job ; then
  build_and_test_on_rbe_gpu
else
  build_and_test_on_rbe_cpu
fi

echo "bazel-testlogs (test results) location:"
find "$KOKORO_ARTIFACTS_DIR" \
  -type l,d -name bazel-testlogs || echo "bazel-testlogs not found"
