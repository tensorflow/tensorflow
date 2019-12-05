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
#
# Installs the latest arduino-cli tool in /tmp/arduino-cli

set -e

cd /tmp

rm -rf arduino-cli*
curl -L -O "https://github.com/arduino/arduino-cli/releases/download/0.4.0/arduino-cli_0.4.0_Linux_64bit.tar.gz"
tar xzf arduino-cli_0.4.0_Linux_64bit.tar.gz

# To use with MacOS, replace the previous two lines with:
# curl -L -O "https://github.com/arduino/arduino-cli/releases/download/0.4.0/arduino-cli_0.4.0_MacOS_64bit.tar.gz"
# tar xzf arduino-cli_0.4.0_MacOS_64bit.tar.gz

/tmp/arduino-cli core update-index
/tmp/arduino-cli core install arduino:mbed
