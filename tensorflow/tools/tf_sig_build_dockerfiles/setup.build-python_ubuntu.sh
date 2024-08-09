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
# setup.build-python_ubuntu.sh: Build python from src for SLES/Ubuntu
#    and install some TF dependencies
# Usage: setup.build-python_ubuntu.sh <pyversion> <requirements.txt>
set -xe

function build_python_from_src() {
    VERSION=$1
    REQUIREMENTS=$2
    _pyver="python${VERSION}"
    local python_map
    declare -A python_map=(
        [python3.8]='3.8.12'
        [python3.9]='3.9.12'
        [python3.10]='3.10.9'
        [python3.11]='3.11.2'
        [python3.12]='3.12.4'
    )
    local _ver=${python_map[$_pyver]}
    wget https://www.python.org/ftp/python/${_ver}/Python-${_ver}.tgz
    tar xvf "Python-${_ver}.tgz" && rm -rf "Python-${_ver}.tgz"
    cd Python-${_ver}/
        ./configure --enable-optimizations
        make -j4
        make altinstall

    ln -sf "/usr/local/bin/python${_ver%.*}" /usr/bin/python3
    ln -sf "/usr/local/bin/python${_ver%.*}" /usr/bin/python
    ln -sf "/usr/local/bin/pip${_ver%.*}" /usr/bin/pip3
    ln -sf "/usr/local/lib/python${_ver%.*}" /usr/lib/tf_python
    cd -
}

if (source /etc/os-release && [[ ${NAME} == SLES ]]); then
    build_python_from_src $1 $2
else
    ##UBUNTU
    source ~/.bashrc
    VERSION=$1
    REQUIREMENTS=$2
    PY_VERSION="python${VERSION}"

    # Install Python build from src dependencies
    # See: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    DEBIAN_FRONTEND=noninteractive apt-get --allow-unauthenticated update
    DEBIAN_FRONTEND=noninteractive apt install -y wget software-properties-common
    DEBIAN_FRONTEND=noninteractive apt-get install build-essential libssl-dev zlib1g-dev \
	                                   libbz2-dev libreadline-dev libsqlite3-dev curl git \
                                           libncursesw5-dev xz-utils tk-dev libxml2-dev \
					   libxmlsec1-dev libffi-dev liblzma-dev
    DEBIAN_FRONTEND=noninteractive apt-get clean all


    build_python_from_src $1 $2

    # Install pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    ln -sf "/usr/local/bin/pip${_ver%.*}" /usr/bin/pip3
    python3 -m pip install --no-cache-dir --upgrade pip
    python3 -m pip install -U setuptools
fi # end of conditional check of various distros

which python3
python3 --version

echo "Install Requirements"
# Disable the cache dir to save image space, and install packages
python3 -m pip install --no-cache-dir -r $REQUIREMENTS -U
python3 -m pip install --no-cache-dir --no-deps tf-keras-nightly
