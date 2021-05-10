#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Install packages required by Python3.6 build

# TODO(amitpatankar): Remove this file once we upgrade to ubuntu:16.04
# docker images for Python 3.6 builds.

# fkrull/deadsnakes is for Python3.6
add-apt-repository -y ppa:fkrull/deadsnakes

apt-get update
apt-get upgrade

# Install python dep
apt-get install python-dev
# Install bz2 dep
apt-get install libbz2-dev
# Install curses dep
apt-get install libncurses5 libncurses5-dev
apt-get install libncursesw5 libncursesw5-dev
# Install readline dep
apt-get install libreadline6 libreadline6-dev
# Install sqlite3 dependencies
apt-get install libsqlite3-dev

set -e

# Install Python 3.6 and dev library
wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tar.xz
tar xvf Python-3.6.1.tar.xz
cd Python-3.6.1

./configure
make altinstall
ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip3

pip3 install --upgrade pip

# Install last working version of setuptools. This must happen before we install
# absl-py, which uses install_requires notation introduced in setuptools 20.5.
pip3 install --upgrade setuptools==39.1.0

pip3 install --upgrade virtualenv

set -e

# Install six.
pip3 install --upgrade absl-py
pip3 install --upgrade six==1.10.0

# Install protobuf.
pip3 install --upgrade protobuf==3.6.1

# Remove obsolete version of six, which can sometimes confuse virtualenv.
rm -rf /usr/lib/python3/dist-packages/six*

# Install numpy, scipy and scikit-learn required by the builds

# numpy needs to be installed from source to fix segfaults. See:
# https://github.com/tensorflow/tensorflow/issues/6968
# This workaround isn't needed for Ubuntu 16.04 or later.
pip3 install --no-binary=:all: --upgrade numpy==1.14.5

pip3 install scipy==1.4.1

pip3 install scikit-learn==0.19.1

# pandas required by `inflow`
pip3 install pandas==0.19.2

pip3 install gnureadline

pip3 install bz2file

# Install recent-enough version of wheel for Python 3.6 wheel builds
pip3 install wheel==0.29.0

pip3 install portpicker

pip3 install werkzeug

pip3 install grpcio

# Eager-to-graph execution needs astor, gast and termcolor:
pip3 install --upgrade astor
pip3 install --upgrade gast
pip3 install --upgrade termcolor

pip3 install --upgrade h5py==3.1.0

# Keras
pip3 install keras_preprocessing==1.0.5

# Estimator
pip3 install tf-estimator-nightly==1.12.0.dev20181203 --no-deps
