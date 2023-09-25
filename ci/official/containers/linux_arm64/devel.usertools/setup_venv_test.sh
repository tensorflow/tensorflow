#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
set -euxo pipefail

# Run this from inside the tensorflow github directory.
# Usage: setup_venv_test.sh venv_and_symlink_name "glob pattern for one wheel file"
# Example: setup_venv_test.sh bazel_pip "/tf/pkg/*.whl"
# 
# This will create a venv with that wheel file installed in it, and a symlink
# in ./venv_and_symlink_name/tensorflow to ./tensorflow. We use this for the
# "pip" tests.

python -m venv /$1
mkdir -p $1
rm -f ./$1/tensorflow
ln -s $(ls /$1/lib) /$1/lib/python3
ln -s ../tensorflow $1/tensorflow
# extglob is necessary for @(a|b) pattern matching
# see "extglob" in the bash manual page ($ man bash)
bash -O extglob -c "/$1/bin/pip install $2"
/$1/bin/pip install -r /usertools/test.requirements.txt
