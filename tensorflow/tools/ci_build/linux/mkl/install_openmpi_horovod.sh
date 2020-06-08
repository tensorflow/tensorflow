#!/usr/bin/env bash
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
# Install OpenMPI, OpenSSH and Horovod during Intel(R) MKL container build
# Usage: install_openmpi_horovod.sh [OPENMPI_VERSION=<openmpi version>] [OPENMPI_DOWNLOAD_URL=<openmpi download url>] [HOROVOD_VERSION=<horovod version>]

set -e

apt-get clean && apt-get update -y

# Set default
OPENMPI_VERSION=${OPENMPI_VERSION:-openmpi-2.1.1}
OPENMPI_DOWNLOAD_URL=${OPENMPI_DOWNLOAD_URL:-https://www.open-mpi.org/software/ompi/v2.1/downloads/${OPENMPI_VERSION}.tar.gz}
HOROVOD_VERSION=${HOROVOD_VERSION:-0.19.1}

# Install Open MPI
echo "Installing OpenMPI version ${OPENMPI_VERSION} ..."
echo "OpenMPI Download url ${OPENMPI_DOWNLOAD_URL} ..."

mkdir /tmp/openmpi
cd /tmp/openmpi
curl -fSsL -O ${OPENMPI_DOWNLOAD_URL}
tar zxf ${OPENMPI_VERSION}.tar.gz
cd ${OPENMPI_VERSION}
./configure --enable-mpirun-prefix-by-default
make -j $(nproc) all
make install
ldconfig
cd /
rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real
echo '#!/bin/bash' > /usr/local/bin/mpirun
echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun
chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Check mpi version
echo 'OpenMPI version:'
mpirun --version

# Install OpenSSH for MPI to communicate between containers
( apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        libnuma-dev \
        openssh-server \
        openssh-client && \        
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ) || \
    ( yum -y update && yum -y install \
            numactl-devel \
            openssh-server \
            openssh-clients && \            
    yum clean all ) || \
    ( echo "Unsupported Linux distribution. Aborting!" && exit 1 )
mkdir -p /var/run/sshd
# Allow OpenSSH to talk to containers without asking for confirmation
cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new
echo " StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new
mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Horovod
HOROVOD_WITH_TENSORFLOW=1
python3 -m pip install --no-cache-dir horovod==${HOROVOD_VERSION}
