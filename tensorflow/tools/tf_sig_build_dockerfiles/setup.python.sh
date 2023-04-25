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
# Usage: setup.python.sh <pyversion> <requirements.txt> <runtime mode>
set -xe


function build_python_from_src() {
    VERSION=$1
    REQUIREMENTS=$2
    local python_map
    declare -A python_map=(
        [python3.8]='3.8.12'
        [python3.9]='3.9.12'
        [python3.10]='3.10.9'
        [python3.11]='3.11.2'
    )
    local _ver=${python_map[$VERSION]}
    wget https://www.python.org/ftp/python/${_ver}/Python-${_ver}.tgz
    tar xvf "Python-${_ver}.tgz" && rm -rf "Python-${_ver}.tgz"
    pushd Python-${_ver}/
	./configure --enable-optimizations
	make altinstall -j16

    ln -sf "/usr/local/bin/python${_ver%.*}" /usr/bin/python3
    ln -sf "/usr/local/bin/pip${_ver%.*}" /usr/bin/pip3
    ln -sf "/usr/local/lib/python${_ver%.*}" /usr/lib/tf_python
    popd
}

if (source /etc/os-release && [[ ${NAME} == SLES ]]); then
    build_python_from_src $1 $2
else

source ~/.bashrc
VERSION=$1
REQUIREMENTS=$2

# Add deadsnakes repo for Python installation
add-apt-repository -y 'ppa:deadsnakes/ppa'

# Install Python packages for this container's version
cat >pythons.txt <<EOF
$VERSION
$VERSION-dev
$VERSION-venv
$VERSION-distutils
EOF
/setup.packages.sh pythons.txt

if [[ $3 ]]; then
    echo "Runtime mode"
else
    echo "Dev mode"
    # Re-link pyconfig.h from x86_64-linux-gnu into the devtoolset directory
    # for any Python version present
    pushd /usr/include/x86_64-linux-gnu
    for f in $(ls | grep python); do
      # set up symlink for devtoolset-9
      rm -f /dt9/usr/include/x86_64-linux-gnu/$f
      ln -s /usr/include/x86_64-linux-gnu/$f /dt9/usr/include/x86_64-linux-gnu/$f
    done
    popd
fi

# Setup links for TensorFlow to compile.
# Referenced in devel.usertools/*.bazelrc
ln -sf /usr/bin/$VERSION /usr/bin/python3
ln -sf /usr/bin/$VERSION /usr/bin/python
ln -sf /usr/lib/$VERSION /usr/lib/tf_python

fi # end of conditional check of various distros


# Python 3.10 include headers fix:
# sysconfig.get_path('include') incorrectly points to /usr/local/include/python
# map /usr/include/python3.10 to /usr/local/include/python3.10
if [[ ! -f "/usr/local/include/$VERSION" ]]; then
  ln -sf /usr/include/$VERSION /usr/local/include/$VERSION
fi

# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install --no-cache-dir --upgrade pip

# Disable the cache dir to save image space, and install packages
python3 -m pip install --no-cache-dir -r $REQUIREMENTS -U
