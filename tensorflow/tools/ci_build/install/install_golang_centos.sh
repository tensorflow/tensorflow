#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

set -ex

GOLANG_URL="https://dl.google.com/go/go1.12.6.linux-amd64.tar.gz"


cd /usr/src
wget "${GOLANG_URL}"
tar -xzf go1.12.6.linux-amd64.tar.gz
mv go /usr/local
rm /usr/src/go1.12.6.linux-amd64.tar.gz