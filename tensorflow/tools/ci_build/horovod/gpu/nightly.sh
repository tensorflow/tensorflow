#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -x

# Source the external common scripts.
source tensorflow/tools/ci_build/release/common.sh

# Exit src directory to avoid Python import issues.
# We do not need TensorFlow source files.
mkdir /tmp/horovod_test
cd /tmp/horovod_test


# Update the latest Python dependency packages via pip3.7
install_ubuntu_16_pip_deps pip3.7

# Install latest bazel
install_bazelisk
which bazel

# Install realpath
sudo apt-get install realpath

# Install tf-nightly and verify version.
pip3.7 install --user --upgrade tf-nightly

python3.7 -c "import tensorflow as tf; print(tf.__version__)"

# Download and install open-mpi.
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz
tar xvf openmpi-4.0.4.tar.gz

# Install gcc.
sudo apt install --assume-yes build-essential

gcc --version

cd openmpi-4.0.4
./configure

# Install open-mpi.
sudo make all install
export LD_LIBRARY_PATH=/usr/local/lib/openmpi
sudo ldconfig

# Install Horovod.
cd ..
HOROVOD_WITH_TENSORFLOW=1
pip3.7 install horovod[tensorflow] --user

# Install tests.
git clone https://github.com/DEKHTIARJonathan/TF_HVD_Stability_Test.git

# Install pytest.
pip3.7 install -U pytest --user

# Install requirements.
cd TF_HVD_Stability_Test
pip3.7 install -r requirements.txt --user

# Run the tests.
python3.7 -m pytest
