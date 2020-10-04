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

install_ubuntu_16_pip_deps pip3.5

install_bazelisk

python2.7 tensorflow/tools/ci_build/update_version.py --nightly

# Remove need to run configure to set TF_NEED_CUDA=0, TF_NEED_ROCM=0, and
# CC_OPT_FLAGS='-mavx' and create .tf_configure.bazelrc with touch and echo.
touch .tf_configure.bazelrc
echo "build --action_env PYTHON_BIN_PATH=\"/usr/local/bin/python3.5\"" >> .tf_configure.bazelrc
echo "build --action_env PYTHON_LIB_PATH=\"/usr/local/lib/python3.5/site-packages\"" >> .tf_configure.bazelrc
echo "build --python_path=\"/usr/local/bin/python3.5\"" >> .tf_configure.bazelrc
echo "build --config=xla" >> .tf_configure.bazelrc
echo "build:opt --copt=-mavx" >> .tf_configure.bazelrc
echo "build:opt --host_copt=-march=native" >> .tf_configure.bazelrc
echo "build:opt --define with_default_optimizations=true" >> .tf_configure.bazelrc
echo "test --flaky_test_attempts=3" >> .tf_configure.bazelrc
echo "test --test_size_filters=small,medium" >> .tf_configure.bazelrc
echo "test:v1 --test_tag_filters=-benchmark-test,-no_oss,-gpu,-oss_serial" >> .tf_configure.bazelrc
echo "test:v1 --build_tag_filters=-benchmark-test,-no_oss,-gpu" >> .tf_configure.bazelrc
echo "test:v2 --test_tag_filters=-benchmark-test,-no_oss,-gpu,-oss_serial,-v1only" >> .tf_configure.bazelrc
echo "test:v2 --build_tag_filters=-benchmark-test,-no_oss,-gpu,-v1only" >> .tf_configure.bazelrc
echo "build --action_env TF_CONFIGURE_IOS=\"0\"" >> .tf_configure.bazelrc


# Build the pip package
bazel build --config=release_cpu_linux tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --cpu --nightly_flag

# Upload the built packages to pypi.
for WHL_PATH in $(ls pip_pkg/tf_nightly_cpu-*dev*.whl); do

  WHL_DIR=$(dirname "${WHL_PATH}")
  WHL_BASE_NAME=$(basename "${WHL_PATH}")
  AUDITED_WHL_NAME="${WHL_DIR}"/$(echo "${WHL_BASE_NAME//linux/manylinux2010}")
  auditwheel repair --plat manylinux2010_x86_64 -w "${WHL_DIR}" "${WHL_PATH}"

  # test the whl pip package
  chmod +x tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh
  ./tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh ${AUDITED_WHL_NAME}
  RETVAL=$?

  # Upload the PIP package if whl test passes.
  if [ ${RETVAL} -eq 0 ]; then
    echo "Basic PIP test PASSED, Uploading package: ${AUDITED_WHL_NAME}"
    twine upload -r pypi-warehouse "${AUDITED_WHL_NAME}"
  else
    echo "Basic PIP test FAILED, will not upload ${AUDITED_WHL_NAME} package"
    return 1
  fi
done
