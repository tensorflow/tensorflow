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
# Tests an individual Arduino library. Because libraries need to be installed
# globally, this can cause problems with previously-installed modules, so we
# recommend that you only run this within a VM.

set -e

PATH=`pwd`/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin:${PATH}
cd ${1}

mbed config root .
mbed deploy

python -c 'import fileinput, glob;
for filename in glob.glob("mbed-os/tools/profiles/*.json"):
  for line in fileinput.input(filename, inplace=True):
    print(line.replace("\"-std=gnu++98\"","\"-std=c++11\", \"-fpermissive\""))'

mbed compile -m DISCO_F746NG -t GCC_ARM
