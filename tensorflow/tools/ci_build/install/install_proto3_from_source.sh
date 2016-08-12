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

# Install protobuf3 from source.

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


# Build and install protobuf.
PROTOBUF_VERSION="3.0.0-beta-2"
PROTOBUF_DOWNLOAD_DIR="/tmp/protobuf"

mkdir "${PROTOBUF_DOWNLOAD_DIR}"
pushd "${PROTOBUF_DOWNLOAD_DIR}"
curl -fSsL -O https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-cpp-$PROTOBUF_VERSION.tar.gz
tar zxf protobuf-cpp-$PROTOBUF_VERSION.tar.gz
cd protobuf-$PROTOBUF_VERSION
./autogen.sh
./configure
make --jobs=${N_JOBS}
sudo make install
make clean
sudo ldconfig
popd
rm -rf "${PROTOBUF_DOWNLOAD_DIR}"
