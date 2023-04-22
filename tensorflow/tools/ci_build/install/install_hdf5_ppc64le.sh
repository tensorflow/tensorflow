#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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


#This is required because pypi doesn't have a pre-built h5py binary for ppc64le
#It has to be compiled from source during the install
apt-get update
apt-get install -y libhdf5-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

#h5py will then be installed from the install_pip_packages.sh script
