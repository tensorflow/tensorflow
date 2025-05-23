#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# setup.python.sh: Install a specific Python version and packages for it.
# Usage: setup.python.sh <pyversion> <requirements.txt>
set -xe

source ~/.bashrc
VERSION=$1
REQUIREMENTS=$2

# Install Python packages for this container's version
if [[ ${VERSION} == "python3.13-nogil" ]]; then
  cat >pythons.txt <<EOF
$VERSION
EOF
elif [[ ${VERSION} == "python3.14" || ${VERSION} == "python3.13" || ${VERSION} == "python3.12" ]]; then
  cat >pythons.txt <<EOF
$VERSION
$VERSION-dev
$VERSION-venv
EOF
else
  cat >pythons.txt <<EOF
$VERSION
$VERSION-dev
$VERSION-venv
$VERSION-distutils
EOF
fi

if [[ ${VERSION} == "python3.14" ]]; then
  # Build python 3.14.0a6 from source now. This is a temporary solution until
  # the astral python standalone build repo release supports for 3.14.0b1.
  # TODO(kanglan): Remove this once python 3.14.0b1 is available in astral
  # python standalone build repo and the rules_python patch is updated.
  apt update && apt install -y libssl-dev zlib1g-dev libbz2-dev libreadline-dev libncurses5-dev libffi-dev liblzma-dev
  wget https://www.python.org/ftp/python/3.14.0/Python-3.14.0a6.tar.xz
  tar -xf Python-3.14.0a6.tar.xz
  pushd Python-3.14.0a6
  mkdir -p /python314-0a6
  CC=clang-18 CXX=clang++-18 ./configure --prefix /python314-0a6 --with-ensurepip=install
  make -j$(nproc)
  make install -j$(nproc)
  ln -s /python314-0a6/bin/python3 /usr/bin/python3.14
  popd
else
  /setup.packages.sh pythons.txt
fi

# Re-link pyconfig.h from x86_64-linux-gnu into the devtoolset directory
# for any Python version present
pushd /usr/include/x86_64-linux-gnu
for f in $(ls | grep python); do
  # set up symlink for devtoolset-9
  rm -f /dt9/usr/include/x86_64-linux-gnu/$f
  ln -s /usr/include/x86_64-linux-gnu/$f /dt9/usr/include/x86_64-linux-gnu/$f
done
popd

# Python 3.10 include headers fix:
# sysconfig.get_path('include') incorrectly points to /usr/local/include/python
# map /usr/include/python3.10 to /usr/local/include/python3.10
if [[ ! -f "/usr/local/include/$VERSION" ]]; then
  ln -sf /usr/include/$VERSION /usr/local/include/$VERSION
fi

# Install pip
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 https://bootstrap.pypa.io/get-pip.py
/usr/bin/$VERSION get-pip.py
/usr/bin/$VERSION -m pip install --no-cache-dir --upgrade pip
/usr/bin/$VERSION -m pip install -U setuptools


# For Python 3.13t, do not install twine as it does not have pre-built wheels
# for this Python version and building it from source fails. We only need twine
# to be present on the system Python which in this case is 3.12.
# Same reason for Python 3.140a7.
if [[ ${VERSION} == "python3.13-nogil" || ${VERSION} == "python3.14" ]]; then
  grep -v "twine" $REQUIREMENTS > requirements_without_twine.txt
  REQUIREMENTS=requirements_without_twine.txt
fi

# Disable the cache dir to save image space, and install packages
/usr/bin/$VERSION -m pip install --no-cache-dir -r $REQUIREMENTS -U
