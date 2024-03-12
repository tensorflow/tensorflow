#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euox pipefail -o history

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

cd "${KOKORO_ARTIFACTS_DIR}/github/xla"

# Install Bazelisk, Bats, Pyenv, Python, upgrade pip, and activate ".tf-venv"
# virtual environment. We use the "PYENV_VERSION" variable here to decide which
# Python version to install. In addition, we update $PATH with the PYENV_ROOT
# environment variable and we set STATIC_DEPS=true for installing lxml for
# Python. Finally, we set up a symlink to the Python packages directory in
# ".tf-venv" which is referenced in macos.bazelrc.
function install_build_env_tools(){
  # Install Bazelisk; Useful as it would automatically pick the correct
  # version of Bazel.
  echo "===== Installing Bazelisk ====="
  sudo wget --no-verbose -O "/usr/local/bin/bazel" \
      "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-darwin-amd64" \
      && chmod +x "/usr/local/bin/bazel"

  echo "===== Installing Pyenv ====="
  # Install pyenv; Set up a virtual environment to control dependencies and their
  # versions
  git clone --branch v2.3.17 https://github.com/pyenv/pyenv.git /Users/kbuilder/.tf_pyenv
  export PYENV_ROOT=/Users/kbuilder/.tf_pyenv
  export PATH="$PYENV_ROOT/bin:$PATH"    # if `pyenv` is not already on PATH
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"

  echo "===== Installing Python ====="
  # Install Python and set the local python version
  pyenv install -s "${TF_PYENV_VERSION}"
  pyenv rehash
  pyenv local "${TF_PYENV_VERSION}"
  # Do a sanity check to make sure that we using the correct Python version
  echo "===== Python version ====="
  python --version
  # Set up virtual environment and activate it
  python -m venv /Users/kbuilder/.tf-venv && source /Users/kbuilder/.tf-venv/bin/activate

  # Setup links to Python. Referenced in ./macos.bazelrc
  ln -s /Users/kbuilder/.tf-venv/lib/python* /Users/kbuilder/.tf-venv/lib/python

  echo "===== Upgrading to latest pip ====="
  python -m pip install --upgrade pip
}

# Run the tests under /Volumes/BuildData/ so that we don't run into VM
# out of space error
mkdir -p /Volumes/BuildData/bazel_output
export TEST_TMPDIR=/Volumes/BuildData/bazel_output

install_build_env_tools

python -m pip install numpy==1.21.4

TARGET_FILTER="-//xla/hlo/experimental/... -//xla/python_api/... -//xla/python/... -//xla/service/gpu/..."
TAGS_FILTER="-no_oss,-oss_excluded,-gpu,-no_mac,-nomac,-mac_excluded"

bazel test \
    --output_filter="" \
    --macos_minimum_os=10.15 \
    --keep_going \
    --test_output=errors \
    --config=nonccl \
    --build_tag_filters=$TAGS_FILTER  --test_tag_filters=$TAGS_FILTER \
    --test_size_filters=small,medium \
    -- //xla/... $TARGET_FILTER
