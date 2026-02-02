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

# Install pip packages from whl files to avoid the time-consuming process of
# building from source.

pip install wheel
pip3 install wheel

# Use pip to install numpy to the latest version, instead of 1.8.2 through
# apt-get
wget -q https://pypi.python.org/packages/17/f3/404bc85be67150663024d2bb5af654c7d16cf678077690dda27b91be14eb/numpy-1.8.2-cp27-cp27mu-manylinux1_x86_64.whl#md5=3ccf5c004fc99bd06dd443de80d622e6
mv numpy-1.8.2-cp27-cp27mu-manylinux1_x86_64.whl \
   numpy-1.8.2-cp27-none-linux_x86_64.whl
pip install numpy-1.8.2-cp27-none-linux_x86_64.whl
rm numpy-1.8.2-cp27-none-linux_x86_64.whl

wget -q https://pypi.python.org/packages/33/7d/46d8905d39f462e0f6d1f38e1d165adc2939b9f91ca800e1cba8ef0c0f24/numpy-1.8.2-cp34-cp34m-manylinux1_x86_64.whl#md5=528b2b555d2b6979f10e444cacc04fc9
mv numpy-1.8.2-cp34-cp34m-manylinux1_x86_64.whl \
   numpy-1.8.2-cp34-none-linux_x86_64.whl
pip3 install numpy-1.8.2-cp34-none-linux_x86_64.whl
rm numpy-1.8.2-cp34-none-linux_x86_64.whl

# Use pip to install scipy to get the latest version, instead of 0.13 through
# apt-get.
# pip install scipy==0.15.1
wget -q https://pypi.python.org/packages/00/0f/060ec52cb74dc8df1a7ef1a524173eb0bcd329110404869b392685cfc5c8/scipy-0.15.1-cp27-cp27mu-manylinux1_x86_64.whl#md5=aaac02e6535742ab02f2075129890714
mv scipy-0.15.1-cp27-cp27mu-manylinux1_x86_64.whl \
   scipy-0.15.1-cp27-none-linux_x86_64.whl
pip install scipy-0.15.1-cp27-none-linux_x86_64.whl
rm scipy-0.15.1-cp27-none-linux_x86_64.whl

# pip3 install scipy==0.15.1
wget -q https://pypi.python.org/packages/56/c5/e0d36aaf719aa02ee3da19151045912e240d145586612e53b5eaa706e1db/scipy-0.15.1-cp34-cp34m-manylinux1_x86_64.whl#md5=d5243b0f9d85f4f4cb62514c82af93d4
mv scipy-0.15.1-cp34-cp34m-manylinux1_x86_64.whl \
   scipy-0.15.1-cp34-cp34m-linux_x86_64.whl
pip3 install scipy-0.15.1-cp34-cp34m-linux_x86_64.whl
rm scipy-0.15.1-cp34-cp34m-linux_x86_64.whl

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

# pandas required by tf.learn/inflow
pip install pandas==0.18.1
pip3 install pandas==0.18.1

# Benchmark tests require the following:
pip install psutil
pip3 install psutil
pip install py-cpuinfo
pip3 install py-cpuinfo

# pylint tests require the following:
pip install pylint
pip3 install pylint

# pep8 tests require the following:
pip install pep8
pip3 install pep8

# tf.mock require the following for python2:
pip install mock
