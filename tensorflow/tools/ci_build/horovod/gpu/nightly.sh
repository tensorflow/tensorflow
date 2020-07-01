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

# Source the external common scripts.
source tensorflow/tools/ci_build/release/common.sh


# Install latest bazel
install_bazelisk
which bazel

# Install realpath
sudo apt-get install realpath

# Update the version string to nightly
if [ -n "${IS_NIGHTLY_BUILD}" ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

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
pip3 install horovod tensorflow

# Install tests.
git clone https://github.com/DEKHTIARJonathan/TF_HVD_Stability_Test.git

# Install pytest.
pip3 install -U pytest

# Install requirements.
cd TF_HVD_Stability_Test
pip3 install -r requirements.txt

# Run the tests.
python3 -m pytest
