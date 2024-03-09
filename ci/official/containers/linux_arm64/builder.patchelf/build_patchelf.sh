#!/bin/bash -eu
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# Need a newer version of patchelf as the installed version is buggy in 20.04
# so get patchelf source from 22.04 ie 'jammy' and build it to avoid dependency
# problems that would occur with a binary package

mkdir -p /patchelf
cd /patchelf
echo deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy universe>>/etc/apt/sources.list
apt-get update
apt-get -y build-dep patchelf/jammy
apt-get -b source patchelf/jammy

# This will leave a .deb file for installation in a later stage
