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
install_bazelisk

# Pick a version of xcode
export DEVELOPER_DIR=/Applications/Xcode_10.3.app/Contents/Developer
sudo xcode-select -s "${DEVELOPER_DIR}"

# Set up and install MacOS pip dependencies.
setup_venv_macos python3.6

# For python3 path on Mac
export PATH=$PATH:/usr/local/bin

./tensorflow/tools/ci_build/update_version.py --nightly

# Run configure.
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=$(which python3.6)
yes "" | "$PYTHON_BIN_PATH" configure.py

# Build the pip package
bazel build --config=release_cpu_macos tensorflow/tools/pip_package:build_pip_package
mkdir pip_pkg
./bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg --cpu --nightly_flag

# Copy and rename to tf_nightly
for f in $(ls pip_pkg/tf_nightly_cpu-*dev*macosx*.whl); do
  copy_to_new_project_name "${f}" tf_nightly python
done

# Upload the built packages to pypi.
for f in $(ls pip_pkg/tf_nightly*dev*macosx*.whl); do

  # test the whl pip package
  chmod +x tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh
  ./tensorflow/tools/ci_build/builds/nightly_release_smoke_test.sh ${f}
  RETVAL=$?

  # Upload the PIP package if whl test passes.
  if [ ${RETVAL} -eq 0 ]; then
    echo "Basic PIP test PASSED, Uploading package: ${f}"
    python -m pip install twine
    python -m twine upload -r pypi-warehouse "${f}"
  else
    echo "Basic PIP test FAILED, will not upload ${f} package"
    return 1
  fi
done
