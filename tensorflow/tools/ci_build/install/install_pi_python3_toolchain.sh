#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

dpkg --add-architecture armhf
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
apt-get update
apt-get install -y libpython3-all-dev:armhf
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
apt-get update
rm -rf /usr/local/bin/bazel
apt-get install -y bazel python3 python3-numpy python3-dev python3-pip

# We're using Ubuntu 14.04 as our base image because that's needed by the Pi
# cross-compilation chain, but that doesn't have built-in Python 3.5 support, so
# install from a separate repository.
apt-get install -y software-properties-common
add-apt-repository ppa:fkrull/deadsnakes
apt-get update
apt-get install -y python3.5 python3.5-dev
