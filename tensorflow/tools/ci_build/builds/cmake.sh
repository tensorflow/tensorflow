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
# ==============================================================================

set -e


# Determine the number of cores, for parallel make.
N_JOBS=$(grep -c ^processor /proc/cpuinfo)
if [[ -z ${N_JOBS} ]]; then
  # The Linux way didn't work. Try the Mac way.
  N_JOBS=$(sysctl -n hw.ncpu)
fi
if [[ -z ${N_JOBS} ]]; then
  N_JOBS=1
  echo ""
  echo "WARNING: Failed to determine the number of CPU cores. "\
"Will use --jobs=1 for make."
fi

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

popd
