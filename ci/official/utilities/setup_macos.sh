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
# those VMs does not support installing Python 3.12 and above which we need
# for running smoke tests in nightly/release wheel builds.
if [[ "${TFCI_MACOS_UPGRADE_PYENV_ENABLE}" == 1 ]]; then
  echo "Upgrading pyenv..."
  echo "Current pyevn version: $(pyenv --version)"

  # Check if pyenv is managed by homebrew. If so, update and upgrade pyenv.
  # Otherwise, install the latest pyenv from github.
  if command -v brew &> /dev/null && brew list pyenv &> /dev/null; then
    # On "ventura-slcn" VMs, pyenv is managed via Homebrew.
    echo "pyenv is installed and managed by homebrew."
    brew update && brew upgrade pyenv
  else
    echo "pyenv is not managed by homebrew. Installing it via github..."
    # On "ventura" VMs, pyenv is not managed by Homebrew. Install the latest
    # pyenv from github.
    rm -rf "$PYENV_ROOT"
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
  fi
  echo "Upgraded pyenv version: $(pyenv --version)"
fi

# "TFCI_MACOS_PYENV_INSTALL_ENABLE" controls whether to use Pyenv to install
# the Python version set in "TFCI_PYTHON_VERSION" and use it as default.
# We enable this in the nightly and release builds because before uploading the
# wheels, we install them in a virtual environment and run some smoke tests on
# it. TFCI Mac VMs only have one Python version installed so we need to install
# the other versions manually.
if [[ "${TFCI_MACOS_PYENV_INSTALL_ENABLE}" == 1 ]]; then
  # Install the necessary Python, unless it's already present
  pyenv install -s "$TFCI_PYTHON_VERSION"
  pyenv local "$TFCI_PYTHON_VERSION"
  # Do a sanity check to make sure that we using the correct Python version
  python --version
fi

# TFCI Mac VM images do not have twine installed by default so we need to
# install it manually. We use Twine in nightly builds to upload Python packages
# to PyPI.
if [[ "$TFCI_MACOS_TWINE_INSTALL_ENABLE" == 1 ]]; then
  pip install twine==3.6.0
fi

# Scheduled nightly and release builds upload build artifacts (Pip packages,
# Libtensorflow archives) to GCS buckets. TFCI Mac VMs need to authenticate as
# a service account that has the right permissions to be able to do so.
set +x
if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  # Python 3.12 removed the module `imp` which is needed by gcloud CLI so we set
  # `CLOUDSDK_PYTHON` to Python 3.11 which is the system Python on TFCI Mac
  # VMs.
  export CLOUDSDK_PYTHON=$(which python3.11)
  gcloud auth activate-service-account --key-file="${GOOGLE_APPLICATION_CREDENTIALS}"
fi
set -x

# When cross-compiling with RBE, we need to copy the macOS sysroot to be
# inside the TensorFlow root directory. We then define them as a filegroup
# target inside "tensorflow/tools/toolchains/cross_compile/cc" so that Bazel
# can register it as an input to compile/link actions and send it to the remote
# VMs when needed.
# TODO(b/316932689): Avoid copying and replace with a local repository rule.
if [[ "$TFCI_MACOS_CROSS_COMPILE_ENABLE" == 1 ]]; then
  mkdir -p "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/usr"
  mkdir -p "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/System/Library/Frameworks/"
  cp -r "${TFCI_MACOS_CROSS_COMPILE_SDK_SOURCE}/usr/include" "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/usr/include"
  cp -r "${TFCI_MACOS_CROSS_COMPILE_SDK_SOURCE}/usr/lib" "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/usr/lib"
  cp -r "${TFCI_MACOS_CROSS_COMPILE_SDK_SOURCE}/System/Library/Frameworks/CoreFoundation.framework" "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/System/Library/Frameworks/CoreFoundation.framework"
  cp -r "${TFCI_MACOS_CROSS_COMPILE_SDK_SOURCE}/System/Library/Frameworks/Security.framework" "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/System/Library/Frameworks/Security.framework"
  cp -r "${TFCI_MACOS_CROSS_COMPILE_SDK_SOURCE}/System/Library/Frameworks/SystemConfiguration.framework" "${TFCI_MACOS_CROSS_COMPILE_SDK_DEST}/System/Library/Frameworks/SystemConfiguration.framework"
fi