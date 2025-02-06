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

add-apt-repository ppa:deadsnakes/ppa
# Install Python packages for this container's version
if [[ ${VERSION} == "python3.13" ]]; then
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
EOF
fi
/setup.packages.sh pythons.txt

# Re-link pyconfig.h from aarch64-linux-gnu into the devtoolset directory
# for any Python version present
pushd /usr/include/aarch64-linux-gnu
for f in $(ls | grep python); do
  # set up symlink for devtoolset-10
  rm -f /dt10/usr/include/aarch64-linux-gnu/$f
  ln -s /usr/include/aarch64-linux-gnu/$f /dt10/usr/include/aarch64-linux-gnu/$f
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

# Disable the cache dir to save image space, and install packages
/usr/bin/$VERSION -m pip install --no-cache-dir -r $REQUIREMENTS -U
