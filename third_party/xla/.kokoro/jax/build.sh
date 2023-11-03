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

clone_main_jax() {
  git clone https://github.com/google/jax.git
}

build_and_test_on_rbe_cpu() {
  export JAX_ENABLE_X64=0

  cd "${KOKORO_ARTIFACTS_DIR}"

  use_local_or_install_python
  install_packages "$NUMPY_VERSION" "$SCIPY_VERSION"
  clone_main_jax
  # Install bazel
  update_bazel_linux

  chmod +x "${KOKORO_GFILE_DIR}/bazel_wrapper.py"
  cd jax

  # Run the tests.
   "${KOKORO_GFILE_DIR}/bazel_wrapper.py" \
      test \
      --verbose_failures=true \
      --override_repository=xla="${KOKORO_ARTIFACTS_DIR}"/github/xla \
      --config=avx_posix \
      --config=tpu \
      --config=mkl_open_source_only \
      --config="$NOCUDA_RBE_CONFIG_NAME" \
      --config=tensorflow_testing_rbe_linux \
      --test_env=JAX_NUM_GENERATED_CASES=25 \
      //tests:cpu_tests //tests:backend_independent_tests \
      --test_output=errors
}

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

# TODO(ddunleavy): move below to different script
NOCUDA_RBE_CONFIG_NAME="rbe_cpu_linux_py312"

setup_env_vars_py312
build_and_test_on_rbe_cpu

echo "bazel-testlogs (test results) location:"
find "$KOKORO_ARTIFACTS_DIR" \
  -type l,d -name bazel-testlogs || echo "bazel-testlogs not found"
