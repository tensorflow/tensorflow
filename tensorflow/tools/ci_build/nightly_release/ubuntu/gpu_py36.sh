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

install_ubuntu_16_python_pip_deps python3.6

install_bazelisk

python2.7 tensorflow/tools/ci_build/update_version.py --nightly

# Run configure.
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python3.6)
yes "" | "$PYTHON_BIN_PATH" configure.py

# Build the pip package
bazel build --config=release_gpu_linux tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --nightly_flag
./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --gpu --nightly_flag

# Upload the built packages to pypi.
for WHL_PATH in $(ls pip_pkg/tf_nightly*dev*.whl); do

  WHL_DIR=$(dirname "${WHL_PATH}")
  WHL_BASE_NAME=$(basename "${WHL_PATH}")
  AUDITED_WHL_NAME="${WHL_DIR}"/$(echo "${WHL_BASE_NAME//linux/manylinux2010}")

  # Copy and rename for gpu manylinux as we do not want auditwheel to package in libcudart.so
  WHL_PATH=${AUDITED_WHL_NAME}
  cp "${WHL_DIR}"/"${WHL_BASE_NAME}" "${WHL_PATH}"
  echo "Copied manylinux2010 wheel file at: ${WHL_PATH}"

  # test the whl pip package
  chmod +x tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh
  ./tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh ${AUDITED_WHL_NAME}
  RETVAL=$?

  # Upload the PIP package if whl test passes.
  if [ ${RETVAL} -eq 0 ]; then
    echo "Basic PIP test PASSED, Uploading package: ${AUDITED_WHL_NAME}"
    # Although the python version installing twine is independent of python
    # version of the binary it pushes, unsure which python versions are
    # available to user.
    python3.6 -m pip install twine
    python3.6 -m twine upload -r pypi-warehouse "${AUDITED_WHL_NAME}"
  else
    echo "Basic PIP test FAILED, will not upload ${AUDITED_WHL_NAME} package"
    return 1
  fi
done
