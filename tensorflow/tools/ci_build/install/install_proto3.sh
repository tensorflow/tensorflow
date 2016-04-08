#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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

# Select bazel version.
PROTOBUF_VERSION="3.0.0-beta-2"

# Install protobuf3.
mkdir /protobuf
cd /protobuf
curl -fSsL -O https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-cpp-$PROTOBUF_VERSION.tar.gz
tar zxf protobuf-cpp-$PROTOBUF_VERSION.tar.gz
cd protobuf-$PROTOBUF_VERSION
./autogen.sh
./configure
make
make install
make clean
ldconfig
cd /; rm -rf /protobuf
