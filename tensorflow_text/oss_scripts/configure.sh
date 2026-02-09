#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

osname="$(uname -s | tr 'A-Z' 'a-z')"
echo $osname

function is_macos() {
  [[ "${osname}" == "darwin" ]]
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Don't install tensorflow locally.
if (which python3) | grep -q "python3"; then
  installed_python="python3"
elif (which python) | grep -q "python"; then
  installed_python="python"
fi

# if [[ $($installed_python -m pip show tensorflow) == *tensorflow* ]] ||
#    [[ $($installed_python -m pip show tensorflow-macos) == *tensorflow-macos* ]] ||
#    [[ $($installed_python -m pip show tf-nightly) == *tf-nightly* ]]; then
#   echo 'Using installed tensorflow.'
# else
#   echo 'Installing tensorflow.'
#   if [[ "$IS_NIGHTLY" == "nightly" ]]; then
#     $installed_python -m pip install --no-deps tf-nightly
#   else
#     $installed_python -m pip install --no-deps tensorflow==2.20
#   fi
# fi

# Copy the current bazelversion of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/v2.20.0/.bazelversion -o .bazelversion

# Copy the building configuration of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/v2.20.0/.bazelrc -o .bazelrc
# This line breaks Windows builds, so we remove it.
sed -i -e 's/build --noincompatible_remove_legacy_whole_archive//' .bazelrc

write_to_bazelrc "build:manylinux2014 --config=release_cpu_linux"
write_to_bazelrc "test --test_env=TF_USE_LEGACY_KERAS=1"

if [ -z "$HERMETIC_PYTHON_VERSION" ]; then
  if [ -n "$PY_VERSION" ]; then
    HERMETIC_PYTHON_VERSION=$PY_VERSION
  else
    HERMETIC_PYTHON_VERSION=$($installed_python  -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
  fi
fi
export HERMETIC_PYTHON_VERSION

# Only auto-upgrade requirements for nightly.
if [[ "$IS_NIGHTLY" == "nightly" ]]; then
  bazel run //oss_scripts/pip_package:requirements.update -- --upgrade
fi

TF_ABIFLAG=$(bazel run //oss_scripts/pip_package:tensorflow_build_info -- abi)
SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.2.dylib"
fi

write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_ABIFLAG}
