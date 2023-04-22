#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Source the external common scripts.
source tensorflow/tools/ci_build/release/common.sh


# Install latest bazel
install_bazelisk
which bazel

# Install realpath
sudo apt-get install realpath

# Update the version string to nightly
if [ -n "${IS_NIGHTLY}" ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

./tensorflow/tools/ci_build/linux/libtensorflow.sh

# Copy the nightly version update script
if [ -n "${IS_NIGHTLY}" ]; then
  cp tensorflow/tools/ci_build/builds/libtensorflow_nightly_symlink.sh lib_package

  echo "This package was built on $(date)" >> lib_package/build_time.txt

  tar -zcvf ubuntu_cpu_libtensorflow_binaries.tar.gz lib_package

  gsutil cp ubuntu_cpu_libtensorflow_binaries.tar.gz gs://libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/cpu
fi

# Upload to go/tf-sizetracker
# TODO(191668861): Re-enable once issue is resolved.
# python3 ./tensorflow/tools/ci_build/sizetrack_helper.py \
#   --team tensorflow_libtensorflow \
#   --artifact_id ubuntu_cpu_nightly \
#   --upload \
#   --artifact "$(find lib_package -iname "libtensorflow*.tar.gz" -not -iname "*jni*" | head -n 1)"
