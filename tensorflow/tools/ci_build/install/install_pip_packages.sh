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

pip install setuptools
pip3 install setuptools

pip install virtualenv
pip3 install virtualenv

pip install sklearn
pip3 install sklearn

pip install scikit-learn
pip3 install scikit-learn

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
