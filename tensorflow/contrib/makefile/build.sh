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
#
# Makefile build

set -e

# Helper functions
die() {
  echo $1
  exit 1
i}

# Make sure that system is OS X
if [[ $(uname) != "Darwin" ]]; then
  die "ERROR: This makefile build requires OS X, which the current system "\
"is not."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=== Downloading dependencies ==="
echo ""

${SCRIPT_DIR}/download_dependencies.sh

echo ""
echo "=== Building host (OS X) copy of protobuf ==="
echo ""

pushd ${SCRIPT_DIR}/downloads/protobuf
./autogen.sh
./configure
make

echo ""
echo "=== Installing host copy of protobuf ==="
echo ""

sudo make install
popd

echo ""
echo "=== Building iOS native version of protobuf ==="
echo ""
tensorflow/contrib/makefile/compile_ios_protobuf.sh

echo ""
echo "=== Building all iOS architectures for TensorFlow ==="
echo ""
tensorflow/contrib/makefile/compile_ios_tensorflow.sh
