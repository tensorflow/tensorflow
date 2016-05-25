#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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

# Install pip packages from whl files to avoid the time-consuming process of
# building from source.

# Use pip to install numpy to the latest version, instead of 1.8.2 through
# apt-get
wget -q https://pypi.python.org/packages/06/92/3c786303889e6246971ad4c48ac2b4e37a1b1c67c0dc2106dc85cb15c18e/numpy-1.11.0-cp27-cp27mu-manylinux1_x86_64.whl#md5=6ffb66ff78c28c55bfa09a2ceee487df
mv numpy-1.11.0-cp27-cp27mu-manylinux1_x86_64.whl \
   numpy-1.11.0-cp27-none-linux_x86_64.whl
pip install numpy-1.11.0-cp27-none-linux_x86_64.whl
rm numpy-1.11.0-cp27-none-linux_x86_64.whl

wget -q https://pypi.python.org/packages/ea/ca/5e48a68be496e6f79c3c8d90f7c03ea09bbb154ea4511f5b3d6c825cefe5/numpy-1.11.0-cp34-cp34m-manylinux1_x86_64.whl#md5=08a002aeffa20354aa5045eadb549361
mv numpy-1.11.0-cp34-cp34m-manylinux1_x86_64.whl \
   numpy-1.11.0-cp34-cp34m-linux_x86_64.whl
pip3 install numpy-1.11.0-cp34-cp34m-linux_x86_64.whl
rm numpy-1.11.0-cp34-cp34m-linux_x86_64.whl

# Use pip to install scipy to get the latest version, instead of 0.13 through
# apt-get.
# pip install scipy==0.17.1
wget -q https://pypi.python.org/packages/8a/de/326cf31a5a3ba0c01c40cdd78f7140b0510ed80e6d5ec5b2ec173c72df03/scipy-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl#md5=8d0df61ceba78a2796f8d90fc979576f
mv scipy-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl \
   scipy-0.17.1-cp27-none-linux_x86_64.whl
pip install scipy-0.17.1-cp27-none-linux_x86_64.whl
rm scipy-0.17.1-cp27-none-linux_x86_64.whl

# pip3 install scipy==0.17.1
wget -q https://pypi.python.org/packages/eb/2e/76aff3b25dd06cab06622f82a4790ff5002ab686e940847bb2503b4b2122/scipy-0.17.1-cp34-cp34m-manylinux1_x86_64.whl#md5=bb39b9e1d16fa220967ad7edd39a8b28
mv scipy-0.17.1-cp34-cp34m-manylinux1_x86_64.whl \
   scipy-0.17.1-cp34-cp34m-linux_x86_64.whl
pip3 install scipy-0.17.1-cp34-cp34m-linux_x86_64.whl
rm scipy-0.17.1-cp34-cp34m-linux_x86_64.whl

# pip install sklearn
wget -q https://pypi.python.org/packages/bf/80/06e77e5a682c46a3880ec487a5f9d910f5c8d919df9aca58052089687c7e/scikit_learn-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl#md5=337b91f502138ba7fd722803138f6dfd
mv scikit_learn-0.17.1-cp27-cp27mu-manylinux1_x86_64.whl \
   scikit_learn-0.17.1-cp27-none-linux_x86_64.whl
pip install scikit_learn-0.17.1-cp27-none-linux_x86_64.whl
rm scikit_learn-0.17.1-cp27-none-linux_x86_64.whl

# pip3 install scikit-learn
wget -q https://pypi.python.org/packages/7e/f1/1cc8a1ae2b4de89bff0981aee904ff05779c49a4c660fa38178f9772d3a7/scikit_learn-0.17.1-cp34-cp34m-manylinux1_x86_64.whl#md5=a722a7372b64ec9f7b49a2532d21372b
mv scikit_learn-0.17.1-cp34-cp34m-manylinux1_x86_64.whl \
   scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl
pip3 install scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl
rm scikit_learn-0.17.1-cp34-cp34m-linux_x86_64.whl

# Benchmark tests require the following:
pip install psutil
pip3 install psutil
pip install py-cpuinfo
pip3 install py-cpuinfo

# pylint tests require the following:
pip install pylint
pip3 install pylint

# Remove packages in /usr/lib/python* that may interfere with packages in
# /usr/local/lib. These packages may get installed inadvertantly with packages
# such as apt-get python-pandas. Their older versions can mask the more recent
# versions installed above with pip and cause test failures.
rm -rf /usr/lib/python2.7/dist-packages/numpy \
       /usr/lib/python2.7/dist-packages/scipy \
       /usr/lib/python3/dist-packages/numpy \
       /usr/lib/python3/dist-packages/scipy
