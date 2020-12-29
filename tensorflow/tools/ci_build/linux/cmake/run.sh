#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================

set -e
set -x

# Run TensorFlow cmake build.
# Clean up, because certain modules, e.g., highwayhash, seem to be sensitive
# to state.
rm -rf build

mkdir -p build
pushd build

cmake -DCMAKE_BUILD_TYPE=Release ../tensorflow/contrib/cmake
# When building do not use all CPUs due to jobs running out of memory.
# TODO(gunan): Figure out why we run out of memory in large GCE instances.
make --jobs 20 tf_python_build_pip_package

virtualenv cmake_test --system-site-packages
source cmake_test/bin/activate

# For older versions of PIP, remove the ABI tag.
# TODO(gunan) get rid of this part once pip is upgraded on all test machines.
WHEEL_FILE_PATH=`ls tf_python/dist/*tensorflow*.whl`
FIXED_WHEEL_PATH=`echo $WHEEL_FILE_PATH | sed -e s/cp27mu/none/`
mv $WHEEL_FILE_PATH $FIXED_WHEEL_PATH

# Install the pip package we just built.
pip install --upgrade $FIXED_WHEEL_PATH

# Run all tests.
ctest -C Release --output-on-failure -j

# Finalize and go back to the initial directory.
deactivate
popd
