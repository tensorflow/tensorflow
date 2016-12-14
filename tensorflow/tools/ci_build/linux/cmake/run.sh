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

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "make will use ${N_JOBS} concurrent job(s)."
echo ""

# Run TensorFlow cmake build.
# Clean up, because certain modules, e.g., highwayhash, seem to be sensitive
# to state.
rm -rf build

mkdir -p build
pushd build

cmake -DCMAKE_BUILD_TYPE=Release ../tensorflow/contrib/cmake
make --jobs=${N_JOBS} all

virtualenv cmake_test --system-site-packages
source cmake_test/bin/activate

# Install the pip package we just built.
pip install --upgrade tf_python/dist/*tensorflow*.whl

# Run all tests.
ctest -C Release --output-on-failure -j ${N_JOBS}

# Finalize and go back to the initial directory.
deactivate
popd
