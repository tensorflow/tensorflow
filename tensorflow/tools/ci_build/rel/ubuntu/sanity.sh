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

# Install latest bazel
source tensorflow/tools/ci_build/release/common.sh
install_bazelisk
which bazel

# We need py3 lint
sudo python3.8 -m pip install pep8

# Install pylint
sudo python3.8 -m pip install setuptools --upgrade
sudo python3.8 -m pip install pylint==2.4.4
python3.8 -m pylint --version

# Run tensorflow sanity checks.
tensorflow/tools/ci_build/ci_sanity.sh
