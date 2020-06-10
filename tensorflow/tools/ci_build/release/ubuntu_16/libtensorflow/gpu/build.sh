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

export TF_NEED_CUDA=1

# Update the version string to nightly
if [ -n "${IS_NIGHTLY_BUILD}" ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

./tensorflow/tools/ci_build/linux/libtensorflow.sh
