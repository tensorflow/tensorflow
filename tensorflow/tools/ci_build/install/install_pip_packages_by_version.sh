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

# Called like install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.10"
PIP="$1"
PIP_INSTALL=("${PIP}" "install" "--prefer-binary" --upgrade)

PYTHON="${PIP/pip/python}"
wget "https://bootstrap.pypa.io/get-pip.py"
"${PYTHON}" "get-pip.py" --force-reinstall
rm "get-pip.py"
"${PYTHON}" -m ensurepip --upgrade

PYTHON_VERSION=$(echo ${PIP##*.})  # only the last number, eg. 10

JAX_PACKAGES=(
  "setuptools"
  "wheel"
  "cloudpickle"
  "colorama>=0.4.4"
  # TODO(phawkins): reenable matplotlib once it makes a NumPy 2.0 compatible
  # release.
  # "matplotlib"
  "pillow>=9.1.0"
  "rich"
  "absl-py"
  "portpicker"
  "six"
  "opt-einsum"
  "auditwheel"
  "typing_extensions"
  "ml_dtypes>=0.4.0"
  "importlib_metadata>=4.6"
  "flatbuffers"
  "build"
)

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
  "libclang"
  "markdown"
  "pandas"
  "packaging"
  "portpicker"
  "protobuf==3.20.3"
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
  "werkzeug"
  "wheel"
)

# Get the latest version of pip so it recognize manylinux2010
"${PIP}" "install" "--upgrade" "pip"
"${PIP}" "install" "--upgrade" "setuptools" "virtualenv"

if [[ "$2" == "jax" ]]; then
  "${PIP_INSTALL[@]}" "${JAX_PACKAGES[@]}"
else
  "${PIP_INSTALL[@]}" "${PACKAGES[@]}"
fi

if [[ "$2" == "jax" ]]; then
  # As of NumPy 2.0, wheels must be built against NumPy 2.0, even if we intend
  # to deploy them against Numpy 1.
  "${PIP_INSTALL[@]}" "scipy>=1.13.1"
  if [[ $((${PYTHON_VERSION} < 13)) ]]; then
    "${PIP_INSTALL[@]}" "numpy~=2.0.0"
  else
    "${PIP_INSTALL[@]}" "numpy~=2.1.0"
  fi
else
  # Special casing by version of Python
  # E.g., numpy supports py3.10 only from 1.21.3
  if [[ ${PYTHON_VERSION} -eq 10 ]]; then
    "${PIP_INSTALL[@]}" "numpy==1.21.3"
  elif [[ ${PYTHON_VERSION} -eq 11 ]]; then
    "${PIP_INSTALL[@]}" "numpy==1.23.4"
  else
    "${PIP_INSTALL[@]}" "numpy==1.19"
  fi
fi

