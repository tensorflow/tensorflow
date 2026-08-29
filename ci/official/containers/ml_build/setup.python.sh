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

if [[ ${VERSION} =~ ^(python3\.[0-9]+)(-nogil|t)?$ ]]; then
  BASE_VERSION="${BASH_REMATCH[1]}"
  NOGIL_SUFFIX="${BASH_REMATCH[2]}"  # e.g., -nogil or t
else
  BASE_VERSION="${VERSION}"
  NOGIL_SUFFIX=""
fi

# Install Python packages for this container's version
cat >pythons.txt <<EOF
$VERSION
$BASE_VERSION-dev
$BASE_VERSION-venv
EOF

PYTHON_PRE_RELEASE_VER="3.15.0rc1"
if [[ ${PYTHON_PRE_RELEASE_VER} =~ ^([0-9]+\.[0-9]+) ]]; then
  PRE_RELEASE_BASE="python${BASH_REMATCH[1]}" # e.g., python3.15
else
  PRE_RELEASE_BASE="python3.15"
fi
if [[ ${PYTHON_PRE_RELEASE_VER} =~ ^([0-9]+\.[0-9]+\.[0-9]+) ]]; then
  PRE_RELEASE_FTP_DIR="${BASH_REMATCH[1]}" # e.g., 3.15.0
else
  PRE_RELEASE_FTP_DIR="3.15.0"
fi

if [[ ${BASE_VERSION} == "${PRE_RELEASE_BASE}" ]]; then
  if [[ ! -d Python-${PYTHON_PRE_RELEASE_VER} ]]; then
    apt-get update && apt-get install -y --no-install-recommends clang-18 libssl-dev zlib1g-dev libbz2-dev libreadline-dev libncurses5-dev libffi-dev liblzma-dev
    wget https://www.python.org/ftp/python/${PRE_RELEASE_FTP_DIR}/Python-${PYTHON_PRE_RELEASE_VER}.tar.xz
    tar -xf Python-${PYTHON_PRE_RELEASE_VER}.tar.xz
  fi
  pushd Python-${PYTHON_PRE_RELEASE_VER}

  if [[ -z "${NOGIL_SUFFIX}" ]]; then
    PREFIX="/python-${PYTHON_PRE_RELEASE_VER}"
    CONFIGURE_ARGS=("--with-ensurepip=install")
  else
    PREFIX="/python-${PYTHON_PRE_RELEASE_VER}-nogil"
    CONFIGURE_ARGS=("--disable-gil" "--with-ensurepip=install")
  fi

  mkdir -p "${PREFIX}"
  CC=clang-18 CXX=clang++-18 ./configure --prefix "${PREFIX}" "${CONFIGURE_ARGS[@]}"
  make -j$(nproc)
  make install -j$(nproc)

  if [[ -z "${NOGIL_SUFFIX}" ]]; then
    ln -s ${PREFIX}/bin/python3 /usr/bin/${PRE_RELEASE_BASE}
  else
    ln -s ${PREFIX}/bin/python3 /usr/bin/${PRE_RELEASE_BASE}-nogil
    ln -s ${PREFIX}/bin/python3 /usr/bin/${PRE_RELEASE_BASE}t
  fi
  popd
else
  /setup.packages.sh pythons.txt
fi

# Python 3.10 include headers fix:
# sysconfig.get_path('include') incorrectly points to /usr/local/include/python
# map /usr/include/python3.10 to /usr/local/include/python3.10
if [[ ! -f "/usr/local/include/$VERSION" ]]; then
  ln -sf /usr/include/$VERSION /usr/local/include/$VERSION
fi

# Install pip
if [[ ${BASE_VERSION} == "python3.9" ]]; then
  GET_PIP_URL="https://bootstrap.pypa.io/pip/3.9/get-pip.py"
else
  GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
fi

wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 "${GET_PIP_URL}" -O get-pip.py
/usr/bin/$VERSION get-pip.py
/usr/bin/$VERSION -m pip install --no-cache-dir --upgrade pip
/usr/bin/$VERSION -m pip install -U setuptools


# For Python 3.13t, do not install twine as it does not have pre-built wheels
# for this Python version and building it from source fails. We only need twine
# to be present on the system Python which in this case is 3.12.
# Same reason for Python 3.14.
if [[ ${VERSION} == "python3.13-nogil" || ${BASE_VERSION} == "${PRE_RELEASE_BASE}" ]]; then
  grep -v "twine" $REQUIREMENTS > requirements_without_twine.txt
  REQUIREMENTS=requirements_without_twine.txt
fi

# Disable the cache dir to save image space, and install packages
/usr/bin/$VERSION -m pip install --no-cache-dir -r $REQUIREMENTS -U

# Verify that the installed Python interpreter can create a venv and bootstrap pip
echo "=== Verifying $VERSION ==="
"/usr/bin/$VERSION" -m venv "/tmp/venv-$VERSION"
"/tmp/venv-$VERSION/bin/pip" list
"/tmp/venv-$VERSION/bin/python" -c "import pip; print(pip.__version__)"
rm -rf "/tmp/venv-$VERSION"
