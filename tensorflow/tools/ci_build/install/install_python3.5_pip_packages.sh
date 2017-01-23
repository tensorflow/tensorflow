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
# Install packages required by Python3.5 build

# TODO(cais): Remove this file once we upgrade to ubuntu:16.04 docker images for
# Python 3.5 builds.

# fkrull/deadsnakes is for Python3.5
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
set -e
# Install Python 3.5 and dev library
apt-get install -y --no-install-recommends python3.5 libpython3.5-dev

# Install pip3.4 and numpy for Python 3.4
# This strange-looking install step is a stopgap measure to make the genrule
# contrib/session_bundle/example:half_plus_two pass. The genrule calls Python
# (via bazel) directly, but calls the wrong version of Python (3.4) because
# bazel does not support specification of Python minor versions yet. So we
# install numpy for Python3.4 here so that the genrule will at least not
# complain about missing numpy. Once we upgrade to 16.04 for Python 3.5 builds,
# this will no longer be necessary.
set +e
pip3_version=$(pip3 --version | grep "python 3.4")
if [[ -z $pip3_version ]]; then
  set -e
  wget -q https://bootstrap.pypa.io/get-pip.py
  python3.4 get-pip.py
  rm -f get-pip.py
fi

NUMPY_VERSION="1.11.0"
numpy_ver_flat=$(echo $NUMPY_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_numpy_ver=$(python3 -c "import numpy; print(numpy.__version__)")
local_numpy_ver_flat=$(echo $local_numpy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_numpy_ver_flat ]]; then
  local_numpy_ver_flat=0
fi
if (( $local_numpy_ver_flat < $numpy_ver_flat )); then
  set -e
  pip3 install --upgrade numpy==${NUMPY_VERSION}
fi

# Install pip3.5
set +e
pip35_version=$(pip3.5 --version | grep "python 3.5")
if [[ -z $pip35_version ]]; then
  set -e
  wget -q https://bootstrap.pypa.io/get-pip.py
  python3.5 get-pip.py
  rm -f get-pip.py
fi

set -e
# Install six.
pip3.5 install --upgrade six==1.10.0

# Install protobuf.
pip3.5 install --upgrade protobuf==3.0.0

# Remove obsolete version of six, which can sometimes confuse virtualenv.
rm -rf /usr/lib/python3/dist-packages/six*

# Install numpy, scipy and scikit-learn required by the builds
pip3.5 install --upgrade numpy

set +e
SCIPY_VERSION="0.17.1"
scipy_ver_flat=$(echo $SCIPY_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_scipy_ver=$(python3.5 -c "import scipy; print(scipy.__version__)")
local_scipy_ver_flat=$(echo $local_scipy_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_scipy_ver_flat ]]; then
  local_scipy_ver_flat=0
fi
if (( $local_scipy_ver_flat < $scipy_ver_flat )); then
  set -e
  wget -q https://pypi.python.org/packages/91/f3/0052c245d53eb5f0e13b7215811e52af3791a8a7d31771605697c28466a0/scipy-0.17.1-cp35-cp35m-manylinux1_x86_64.whl#md5=8e77756904c81a6f79ed10e3abf0c544
  pip3.5 install --upgrade scipy-0.17.1-cp35-cp35m-manylinux1_x86_64.whl
  rm -f scipy-0.17.1-cp35-cp35m-manylinux1_x86_64.whl
fi

set -e
pip3.5 install --upgrade scikit-learn

# Install recent-enough version of wheel for Python 3.5 wheel builds
pip3.5 install wheel==0.29.0

pip3.5 install --upgrade pandas==0.18.1

pip3.5 install portpicker
