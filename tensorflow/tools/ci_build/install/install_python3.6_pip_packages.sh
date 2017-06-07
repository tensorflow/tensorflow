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

# Remove obsolete version of six and setup, which can sometimes confuse virtualenv
# and installation of pip3.6.
rm -rf /usr/lib/python3/dist-packages/six /usr/lib/python3/dist-packages/setuptools

# fkrull/deadsnakes is for Python3.6
add-apt-repository -y ppa:fkrull/deadsnakes
apt-get update

set +e
# Upgrade swig to 3.0.8
SWIG_VERSION="3.0.8"
swig_ver_flat=$(echo $SWIG_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_swig_ver=$(swig -version | grep -i version | awk '{print $3}')
local_swig_ver_flat=$(echo $local_swig_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_swig_ver_flat ]]; then
  local_swig_ver_flat=0
fi
if (( $local_swig_ver_flat < $swig_ver_flat )); then
  set -e
  wget -q http://downloads.sourceforge.net/swig/swig-3.0.8.tar.gz
  tar xzf swig-3.0.8.tar.gz
  pushd swig-3.0.8
  apt-get install -y --no-install-recommends libpcre3-dev
  ./configure
  make
  make install
  rm -f /usr/bin/swig
  ln -s /usr/local/bin/swig /usr/bin/swig
  popd
  rm -rf swig-3.0.8 swig-3.0.8.tar.gz
fi

# set -e  # TODO(cais): Confirm removal.

# Install Python 3.6 and dev library
apt-get install -y --no-install-recommends python3.6 libpython3.6-dev

# Install pip3.6
wget -q https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py
rm -f get-pip.py

# Install six.
pip3.6 install --upgrade six==1.10.0

# Install protobuf.
pip3.6 install --upgrade protobuf==3.2.0

# Install numpy, scipy and scikit-learn required by the builds

# numpy needs to be installed from source to fix segfaults. See:
# https://github.com/tensorflow/tensorflow/issues/6968
# This workaround isn't needed for Ubuntu 16.04 or later.
pip3.6 install --no-binary=:all: --upgrade numpy==1.12.0

pip3.6 install scipy==0.18.1

pip3.6 install scikit-learn==0.18.1

# pandas required by tf.learn/inflow
pip3.6 install pandas==0.19.2

# Install recent-enough version of wheel for Python 3.6 wheel builds
pip3.6 install wheel==0.29.0

pip3.6 install portpicker

pip3.6 install werkzeug

pip3.6 install --upgrade virtualenv==15.1.0
