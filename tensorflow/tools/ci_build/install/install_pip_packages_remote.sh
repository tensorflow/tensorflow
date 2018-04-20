#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

if [ ! -f /usr/bin/x86_64-linux-gnu-gcc ]; then
  ln -s /usr/local/bin/clang /usr/bin/x86_64-linux-gnu-gcc
fi

pip2 install --upgrade setuptools
pip3 install --upgrade setuptools

# The rest of the pip packages will be installed in
# `install_pip_packages.sh`
