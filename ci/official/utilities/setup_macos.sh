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

# "TFCI_MACOS_BAZEL_TEST_DIR_PATH" specifies the directory that Bazel should use
# when running tests. Each test will be executed in a separate subdirectory
# inside this directory. TF Mac builds need ~150 GB of disk space to be able to
# run all the tests. Since TFCI Mac VMs execute Bazel test commands in a
# partition with insufficient storage, we specify the
# 'TFCI_MACOS_BAZEL_TEST_DIR_PATH' environment variable to point to a partition
# with ample storage. When this variable is empty (i.e by default), Bazel will
# use the output base directory to run tests.
if [[ "${TFCI_MACOS_BAZEL_TEST_DIR_ENABLE}" == 1 ]]; then
  mkdir -p "${TFCI_MACOS_BAZEL_TEST_DIR_PATH}"
  export TEST_TMPDIR="${TFCI_MACOS_BAZEL_TEST_DIR_PATH}"
fi

# "TFCI_MACOS_INSTALL_BAZELISK_ENABLE" is used to decide if we need to install
# Bazelisk manually. We enable this for macOS x86 builds as those VMs do not
# have Bazelisk pre-installed. "TFCI_MACOS_INSTALL_BAZELISK_URL" contains the
# link to the Bazelisk binary which needs to be downloaded.
if [[ "${TFCI_MACOS_INSTALL_BAZELISK_ENABLE}" == 1 ]]; then
  sudo wget --no-verbose -O "/usr/local/bin/bazel" "${TFCI_MACOS_INSTALL_BAZELISK_URL}"
  chmod +x "/usr/local/bin/bazel"
fi

# "TFCI_MACOS_UPGRADE_PYENV_ENABLE" is used to decide if we need to upgrade the
# Pyenv version. We enable this for macOS x86 builds as the default Pyenv on
# those VMs does not support installing Python 3.10 and above which we need
# for running smoke tests in nightly/release wheel builds.
if [[ "${TFCI_MACOS_UPGRADE_PYENV_ENABLE}" == 1 ]]; then
  brew upgrade pyenv
fi

# "TFCI_MACOS_PYENV_INSTALL_ENABLE" controls whether to use Pyenv to install
# the Python version set in "TFCI_PYTHON_VERSION" and use it as default.
# We enable this in the nightly and release builds because before uploading the
# wheels, we install them in a virtual environment and run some smoke tests on
# it. TFCI Mac VMs only have one Python version installed so we need to install
# the other versions manually.
if [[ "${TFCI_MACOS_PYENV_INSTALL_ENABLE}" == 1 ]]; then
  pyenv install "$TFCI_PYTHON_VERSION"
  pyenv local "$TFCI_PYTHON_VERSION"
  # Do a sanity check to make sure that we using the correct Python version
  python --version
fi

if [[ "$TFCI_PYTHON_VERSION" == "3.12" ]]; then
  # dm-tree (Keras v3 dependency) doesn't have pre-built wheels for 3.12 yet.
  # Having CMake allows building them.
  # Once the wheels are added, this should be removed - b/308399490.
  brew install cmake
fi

# Scheduled nightly and release builds upload build artifacts (Pip packages,
# Libtensorflow archives) to GCS buckets. TFCI Mac VMs need to authenticate as
# a service account that has the right permissions to be able to do so.
set +x
if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  gcloud auth activate-service-account --key-file="${GOOGLE_APPLICATION_CREDENTIALS}"
fi
set -x