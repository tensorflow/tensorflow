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

cd /usr/src
wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
tar xzf Python-3.6.8.tgz
cd Python-3.6.8
./configure --enable-optimizations
make altinstall
rm /usr/src/Python-3.6.8.tgz

# Link the pip3.6 executable to pip3.
ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3
