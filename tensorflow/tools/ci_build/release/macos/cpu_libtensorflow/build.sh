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
echo "chmod go+w lib_package/*" >> tensorflow/tools/ci_build/linux/libtensorflow.sh
echo "bazel clean --expunge" >> tensorflow/tools/ci_build/linux/libtensorflow.sh

# Install latest bazel
source tensorflow/tools/ci_build/release/common.sh
install_bazelisk

# Pick a version of xcode
export DEVELOPER_DIR=/Applications/Xcode_10.3.app/Contents/Developer
sudo xcode-select -s "${DEVELOPER_DIR}"

# Update the version string to nightly
./tensorflow/tools/ci_build/update_version.py --nightly

tensorflow/tools/ci_build/osx/libtensorflow_cpu.sh

# Copy the nightly version update script
cp tensorflow/tools/ci_build/builds/libtensorflow/libtensorflow_nightly_symlink.sh lib_package
