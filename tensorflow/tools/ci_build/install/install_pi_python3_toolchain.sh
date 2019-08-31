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
echo 'deb [arch=armhf] http://ports.ubuntu.com/ xenial main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ xenial-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ xenial-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
apt-get update
apt-get install -y libpython3-all-dev:armhf
apt-get install -y python3 python3-numpy python3-dev python3-pip
