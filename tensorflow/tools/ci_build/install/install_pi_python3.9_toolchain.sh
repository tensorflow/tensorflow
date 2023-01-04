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

yes | add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.9 python3.9-dev
apt-get install -y python3-pip 
ln -sf /usr/bin/python3.9 /usr/local/bin/python3.9
apt-get install -y python3.9-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
update-alternatives --set python3 /usr/bin/python3.9
pip3 install --upgrade pip
# python3.9 -m pip install --upgrade pip
source /install/common.sh
install_ubuntu_16_python_pip_deps python3.9
cp -r /root//.local/lib/python3.9/site-packages/* /usr/lib/python3/dist-packages/.
ln -sf /root//.local/lib/python3.9/site-packages/numpy/core/include/numpy /usr/include/python3.9/numpy 
rm -f /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3
