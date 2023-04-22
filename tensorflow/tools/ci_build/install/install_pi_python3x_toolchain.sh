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

PYTHON_VERSION=$1
dpkg --add-architecture armhf
dpkg --add-architecture arm64
echo 'deb [arch=arm64,armhf] http://ports.ubuntu.com/ xenial main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64,armhf] http://ports.ubuntu.com/ xenial-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64,armhf] http://ports.ubuntu.com/ xenial-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64,armhf] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
yes | add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
#/usr/local/bin/python3.x is needed to use /install/install_pip_packages_by_version.sh
ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python${PYTHON_VERSION}
apt-get install -y libpython${PYTHON_VERSION}-dev:armhf
apt-get install -y libpython${PYTHON_VERSION}-dev:arm64

if [[ "${PYTHON_VERSION}" == "3.8" ]]; then
  apt-get install -y python${PYTHON_VERSION}-distutils
fi

/install/install_pip_packages_by_version.sh "/usr/local/bin/pip${PYTHON_VERSION}"
ln -sf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/numpy/core/include/numpy /usr/include/python${PYTHON_VERSION}/numpy
