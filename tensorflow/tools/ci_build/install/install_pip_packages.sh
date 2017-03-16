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

# We don't apt-get install so that we can install a newer version of pip. Not
# needed after we upgrade to Ubuntu 16.04
easy_install -U pip
easy_install3 -U pip

# Install pip packages from whl files to avoid the time-consuming process of
# building from source.

pip2 install wheel
pip3 install wheel

# Install six.
pip2 install --upgrade six==1.10.0
pip3 install --upgrade six==1.10.0

# Install werkzeug.
pip2 install --upgrade werkzeug==0.11.10
pip3 install --upgrade werkzeug==0.11.10

# Install protobuf.
pip2 install --upgrade protobuf==3.2.0
pip3 install --upgrade protobuf==3.2.0

# Remove obsolete version of six, which can sometimes confuse virtualenv.
rm -rf /usr/lib/python3/dist-packages/six*

# numpy needs to be installed from source to fix segfaults. See:
# https://github.com/tensorflow/tensorflow/issues/6968
# This workaround isn't needed for Ubuntu 16.04 or later.
pip2 install --no-binary=:all: --upgrade numpy==1.12.0
pip3 install --no-binary=:all: --upgrade numpy==1.12.0

pip2 install scipy==0.18.1
pip3 install scipy==0.18.1

pip2 install scikit-learn==0.18.1
pip3 install scikit-learn==0.18.1

# pandas required by tf.learn/inflow
pip2 install pandas==0.19.2
pip3 install pandas==0.19.2

# Benchmark tests require the following:
pip2 install psutil
pip3 install psutil
pip2 install py-cpuinfo
pip3 install py-cpuinfo

# pylint tests require the following:
pip2 install pylint
pip3 install pylint

# pep8 tests require the following:
pip2 install pep8
pip3 install pep8

# tf.mock require the following for python2:
pip2 install mock

pip2 install portpicker
pip3 install portpicker
