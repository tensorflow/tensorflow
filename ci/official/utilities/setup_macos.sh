#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#
# macOS specific setup for all TF scripts.
#

# Mac version of Core utilities differ in usage. Since our scripts are written
# with the GNU style, we need to set GNU utilities to be default on Mac.
if [[ -n "$(which grealpath)" ]] &&  [[ -n "$(which gstat)" ]]; then
  alias realpath=grealpath
  alias stat=gstat
  # By default, aliases are only expanded in interactive shells, which means
  # that they are not substituted for their corresponding commands in shell
  # scripts. By setting "expand_aliases", we enable alias expansion in
  # non-interactive shells as well.
  shopt -s expand_aliases
else
  echo '==TFCI==: Error: Cannot find path to grealpath or gstat'
  echo 'TF CI scripts require GNU core utilties to be installed. Please make'
  echo 'sure they are present on your system and try again.'
  exit 1
fi

if [[ -n "${KOKORO_JOB_NAME}" ]]; then
  # Mac builds need ~150 GB of disk space to be able to run all the tests. By
  # default, Kokoro runs the Bazel commands in a partition that does not have
  # enough free space so we need to set TEST_TMPDIR explicitly.
  mkdir -p /Volumes/BuildData/bazel_output
  export TEST_TMPDIR=/Volumes/BuildData/bazel_output

  # Before uploading the nightly and release wheels, we install them in a
  # virtual environment and run some smoke tests on it. The Kokoro Mac VMs
  # only have Python 3.11 installed so we need to install the other Python
  # versions manually.
  if [[ -n "${TFCI_BUILD_PIP_PACKAGE_ARGS}" ]] && [[ "${TFCI_PYENV_INSTALL_LOCAL_ENABLE}" != 3.11 ]]; then
    pyenv install "${TFCI_PYENV_INSTALL_LOCAL_ENABLE}"
    pyenv local "${TFCI_PYENV_INSTALL_LOCAL_ENABLE}"
  fi
elif [[ "${TFCI_WHL_BAZEL_TEST_ENABLE}" == 1 ]]; then
  echo '==TFCI==: Note: Mac builds need ~150 GB of disk space to be able to'
  echo 'run all the tests. Please make sure your system has enough disk space'
  echo 'You can control where Bazel stores test artifacts by setting the'
  echo '`TEST_TMPDIR` environment variable.'
fi

if [[ "${TFCI_PYTHON_VERSION}" == "3.12" ]]; then
  # dm-tree (Keras v3 dependency) doesn't have pre-built wheels for 3.12 yet.
  # Having CMake allows building them.
  # Once the wheels are added, this should be removed - b/308399490.
  sudo apt-get install -y --no-install-recommends cmake
fi
