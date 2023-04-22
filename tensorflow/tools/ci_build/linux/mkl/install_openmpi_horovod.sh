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
# Usage: install_openmpi_horovod.sh [OPENMPI_VERSION=<openmpi version>] [OPENMPI_DOWNLOAD_URL=<openmpi download url>] 
# [HOROVOD_VERSION=<horovod version>]

set -e

# Set default
OPENMPI_VERSION=${OPENMPI_VERSION:-openmpi-2.1.1}
OPENMPI_DOWNLOAD_URL=${OPENMPI_DOWNLOAD_URL:-https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz}
INSTALL_HOROVOD_FROM_COMMIT=${INSTALL_HOROVOD_FROM_COMMIT:-no}
BUILD_SSH=${BUILD_SSH:-no}
HOROVOD_VERSION=${HOROVOD_VERSION:-0.19.1}
SSH_CONFIG_PATH=/etc/ssh

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
if [[ ${BUILD_SSH} == "yes" ]]; then
	mkdir /tmp/buildssh
	cd /tmp/buildssh && curl -fSsL -O http://www.zlib.net/zlib-1.2.11.tar.gz && tar -xzvf zlib-1.2.11.tar.gz && \
		cd /tmp/buildssh/zlib-1.2.11 && ./configure && make && make install
	cd /tmp/buildssh && curl -fSsL -O https://www.openssl.org/source/openssl-1.1.1.tar.gz && tar -xzvf openssl-1.1.1.tar.gz && \
		cd  /tmp/buildssh/openssl-1.1.1 && ./config && make  && make test  && make install
	cd /tmp/buildssh && curl -fSsL -O https://mirrors.sonic.net/pub/OpenBSD/OpenSSH/portable/openssh-8.4p1.tar.gz && \
		tar -xzvf openssh-8.4p1.tar.gz && cd /tmp/buildssh/openssh-8.4p1 && \
		./configure --with-md5-passwords  && make && \
		groupadd sshd && useradd -M -g sshd -c 'sshd privsep' -d /var/empty -s /sbin/nologin sshd && passwd -l sshd && \
		make install
	apt-get clean && apt-get update && \
	    apt-get install -y --no-install-recommends --fix-missing \
	        libnuma-dev cmake
        SSH_CONFIG_PATH=/usr/local/etc
else
	apt-get clean && apt-get update && \
	    apt-get install -y --no-install-recommends --fix-missing \
	        openssh-client openssh-server libnuma-dev cmake && \
	    rm -rf /var/lib/apt/lists/*
	if [[ $?  == "0" ]]; then
	    echo "PASS: OpenSSH installation"
	else
	    yum -y update && yum -y install numactl-devel openssh-server openssh-clients cmake && \
	        yum clean all
	    if [[ $?  == "0" ]]; then
	        echo "PASS: OpenSSH installation"
	    else
	        echo "Unsupported Linux distribution. Aborting!" && exit 1
	    fi
	fi
fi
mkdir -p /var/run/sshd
grep -v StrictHostKeyChecking ${SSH_CONFIG_PATH}/ssh_config > ${SSH_CONFIG_PATH}/ssh_config.new
# Allow OpenSSH to talk to containers without asking for confirmation
echo " StrictHostKeyChecking no" >> ${SSH_CONFIG_PATH}/ssh_config.new
mv ${SSH_CONFIG_PATH}/ssh_config.new ${SSH_CONFIG_PATH}/ssh_config

# Install Horovod
if [[ ${INSTALL_HOROVOD_FROM_COMMIT} == "yes" ]]; then
	HOROVOD_WITH_TENSORFLOW=1
	python3 -m pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
else
	HOROVOD_WITH_TENSORFLOW=1
	python3 -m pip install --no-cache-dir horovod==${HOROVOD_VERSION}
fi
