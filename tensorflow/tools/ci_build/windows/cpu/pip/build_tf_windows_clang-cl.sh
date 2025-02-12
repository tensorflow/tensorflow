#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.
# change

# All commands shall pass, and all should be visible.
set -x
set -e

# This script is under <repo_root>/tensorflow/tools/ci_build/windows/cpu/pip/
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tensorflow/tools/ci_build/windows/cpu/pip}.

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

# Recreate an empty bazelrc file under source root
export TMP_BAZELRC=.tmp.bazelrc
rm -f "${TMP_BAZELRC}"
touch "${TMP_BAZELRC}"

function cleanup {
  # Remove all options in .tmp.bazelrc
  echo "" > "${TMP_BAZELRC}"
}
trap cleanup EXIT

PY_TEST_DIR="py_test_dir"

SKIP_TEST=0
RELEASE_BUILD=0
TEST_TARGET="//${PY_TEST_DIR}/tensorflow/python/..."
PROJECT_NAME=""
EXTRA_BUILD_FLAGS=""
EXTRA_TEST_FLAGS=""

# --skip_test            Skip running tests
# --enable_remote_cache  Add options to enable remote cache for build and test
# --release_build        Build for release, compilation time will be longer to
#                        ensure performance
# --test_core_only       Use tensorflow/python/... as test target
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf_nightly) TF_NIGHTLY=1 ;;
    --skip_test) SKIP_TEST=1 ;;
    --enable_remote_cache) set_remote_cache_options ;;
    --release_build) RELEASE_BUILD=1 ;;
    --test_core_only) TEST_TARGET="//${PY_TEST_DIR}/tensorflow/python/..." ;;
    --extra_build_flags)
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      EXTRA_BUILD_FLAGS="$1"
      ;;
    --project_name)
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
      ;;
    --extra_test_flags)
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      EXTRA_TEST_FLAGS="$1"
      ;;
    *)
  esac
  shift
done

if [[ "$RELEASE_BUILD" == 1 ]]; then
  # Overriding eigen strong inline speeds up the compiling of conv_grad_ops_3d.cc and conv_ops_3d.cc
  # by 20 minutes. See https://github.com/tensorflow/tensorflow/issues/10521
  # Because this hurts the performance of TF, we don't override it in release build.
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=0
else
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
fi

if [[ "$TF_NIGHTLY" == 1 ]]; then
  if [[ ${PROJECT_NAME} == *"2.0_preview"* ]]; then
    python tensorflow/tools/ci_build/update_version.py --version=2.0.0 --nightly
  else
    python tensorflow/tools/ci_build/update_version.py --nightly
  fi
  if [ -z ${PROJECT_NAME} ]; then
    EXTRA_PIP_FLAGS="--nightly_flag"
  else
    EXTRA_PIP_FLAGS="--project_name ${PROJECT_NAME} --nightly_flag"
  fi
else
  if [[ -v PROJECT_NAME  ]]; then
    EXTRA_PIP_FLAGS="--project_name ${PROJECT_NAME}"
  fi
fi

# Enable short object file path to avoid long path issue on Windows.
echo "startup --output_user_root=${TMPDIR}" >> "${TMP_BAZELRC}"

if ! grep -q "import %workspace%/${TMP_BAZELRC}" .bazelrc; then
  echo "import %workspace%/${TMP_BAZELRC}" >> .bazelrc
fi

run_configure_for_cpu_build

bazel build ${EXTRA_BUILD_FLAGS}  \
  --experimental_cc_shared_library \
  --build_tag_filters=-no_pip,-no_windows,-windows_excluded,-no_oss,-oss_excluded,-gpu,-tpu \
  --config=win_clang \
  --output_filter=^$ \
  tensorflow/lite:framework tensorflow/lite/examples/minimal:minimal || exit $?

bazel build \
  --experimental_cc_shared_library \
  --config=release_cpu_windows ${EXTRA_BUILD_FLAGS} \
  --config=win_clang \
  --output_filter=^$ \
  tensorflow/tools/pip_package:wheel || exit $?

if [[ "$SKIP_TEST" == 1 ]]; then
  exit 0
fi

# Create a python test directory to avoid package name conflict
create_python_test_dir "${PY_TEST_DIR}"

cp ./bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl "$PWD/${PY_TEST_DIR}"

if [[ "$TF_NIGHTLY" == 1 ]]; then
  exit 0
fi

