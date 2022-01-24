#!/bin/bash -eu
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

PIP="$1"
PIP_INSTALL=("${PIP}" "install" "--prefer-binary" --upgrade)

PYTHON="${PIP/pip/python}"
wget "https://bootstrap.pypa.io/get-pip.py"
"${PYTHON}" "get-pip.py" --force-reinstall
rm "get-pip.py"
"${PYTHON}" -m ensurepip --upgrade

PACKAGES=(
  "absl-py"
  "argparse"
  "astor"
  "auditwheel"
  "bleach"
  "dill"
  "dm-tree"
  "future"
  "gast"
  "grpcio"
  "h5py"
  "keras-nightly"
  "keras_preprocessing"
  "libclang"
  "markdown"
  "numpy"
  "pandas"
  "portpicker"
  "protobuf"
  "psutil"
  "py-cpuinfo"
  "pybind11"
  "pycodestyle"
  "pylint==2.7.4"
  "scikit-learn"
  "scipy"
  "six"
  "tb-nightly"
  "tblib"
  "termcolor"
  "tf-estimator-nightly"
  "werkzeug"
  "wheel"
)

# Get the latest version of pip so it recognize manylinux2010
"${PIP}" "install" "--upgrade" "pip"
"${PIP}" "install" "--upgrade" "setuptools" "virtualenv"

"${PIP_INSTALL[@]}" "${PACKAGES[@]}"
