#!/bin/bash -eu
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

VERSION="$1"
NO_RC_VERSION="${VERSION%rc*}"

shift

mkdir /build
cd /build
wget "https://www.python.org/ftp/python/${NO_RC_VERSION}/Python-${VERSION}.tgz"
tar xvzf "Python-${VERSION}.tgz"
cd "Python-${VERSION}"
./configure "$@"
make -j$(nproc) altinstall

rm -rf /build
