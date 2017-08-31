#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

sudo pip3 install auditwheel==1.5.0

set +e
patchelf_location=$(which patchelf)
if [[ -z "$patchelf_location" ]]; then
  set -e
  # Install patchelf from source (it does not come with trusty package)
  wget https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.bz2
  tar xfa patchelf-0.9.tar.bz2
  cd patchelf-0.9
  ./configure --prefix=/usr/local
  make
  sudo make install
fi
cd ..
