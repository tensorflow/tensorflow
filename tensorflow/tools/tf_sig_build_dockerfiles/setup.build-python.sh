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
PYTHON_VERSION=$1
REQUIREMENTS=$2

# PYTHON
if [ "$PYTHON_VERSION" = "3.7" ]; then
    wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz && tar xvf Python-3.7.11.tgz && cd Python-3.7*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.7 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    NUMPY_VERSION=1.18.5
elif [ "$PYTHON_VERSION" = "3.8" ]; then
    wget https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz && tar xvf Python-3.8.9.tgz && cd Python-3.8*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.8 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.8 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    NUMPY_VERSION=1.18.5
elif [ "$PYTHON_VERSION" = "3.9" ]; then
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && tar xvf Python-3.9.7.tgz && cd Python-3.9*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    NUMPY_VERSION=1.20.3
elif [ "$PYTHON_VERSION" = "3.10" ]; then
    #install openssl1.1.1
    wget --no-check-certificate https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz && tar xvf openssl-1.1.1k.tar.gz && cd openssl-1.1.1k &&
        ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic && make && make install

    wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && tar xvf Python-3.10.2.tgz && cd Python-3.10*/ &&
        sed -i 's/PKG_CONFIG openssl /PKG_CONFIG openssl11 /g' configure && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    NUMPY_VERSION=1.21.4
else
    printf '%s\n' "Python Version not Supported" >&2
    exit 1
fi

# Setup links for TensorFlow to compile.
# Referenced in devel.usertools/*.bazelrc
ln -sf /usr/local/bin/$VERSION /usr/bin/python3
ln -sf /usr/local/bin/$VERSION /usr/bin/python
ln -sf /usr/local/lib/$VERSION /usr/lib/tf_python

export PYTHON_LIB_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages
export PYTHON_BIN_PATH=/usr/local/bin/python${PYTHON_VERSION}

# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install --no-cache-dir --upgrade pip

# Disable the cache dir to save image space, and install packages
python3 -m pip install --no-cache-dir -r $REQUIREMENTS -U
