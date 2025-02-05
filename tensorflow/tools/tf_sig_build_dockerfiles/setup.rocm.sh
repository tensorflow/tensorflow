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
# setup.rocm.sh: Prepare the ROCM installation on the container.
# Usage: setup.rocm.sh <ROCM_VERSION> <DISTRO>
# Supported Distros:
#   - focal
#   - jammy
#   - el7
#   - el8
set -x

# Get arguments (or defaults)
ROCM_VERSION=6.2.0
DISTRO=focal
if [[ -n $1 ]]; then
    ROCM_VERSION=$1
fi
if [[ -n $2 ]]; then
    if [[ "$2" == "focal" ]] || [[ "$2" == "jammy" ]] || [[ "$2" == "noble" ]] || [[ "$2" == "el7" ]] || [[ "$2" == "el8" ]]; then
        DISTRO=$2
    else
        echo "Distro not supported"
        echo "Supported distros are:\n focal\n jammy\n noble\n el7\n el8"
	exit 1
    fi
fi

ROCM_PATH=${ROCM_PATH:-/opt/rocm-${ROCM_VERSION}}
# Intial release don't have the trialing '.0'
# For example ROCM 5.4.0 is at https://repo.radeon.com/rocm/apt/5.4/
if [ ${ROCM_VERSION##*[^0-9]} -eq '0' ]; then
        ROCM_VERS=${ROCM_VERSION%.*}
else
        ROCM_VERS=$ROCM_VERSION
fi

if [[ "$DISTRO" == "focal" ]] || [[ "$DISTRO" == "jammy" ]] || [[ "$DISTRO" == "noble" ]]; then
    ROCM_DEB_REPO_HOME=https://repo.radeon.com/rocm/apt/
    AMDGPU_DEB_REPO_HOME=https://repo.radeon.com/amdgpu/
    ROCM_BUILD_NAME=${DISTRO}
    ROCM_BUILD_NUM=main

    # Adjust the ROCM repo location
    ROCM_DEB_REPO=${ROCM_DEB_REPO_HOME}${ROCM_VERS}/
    AMDGPU_DEB_REPO=${AMDGPU_DEB_REPO_HOME}${ROCM_VERS}/

    DEBIAN_FRONTEND=noninteractive apt-get --allow-unauthenticated update 
    DEBIAN_FRONTEND=noninteractive apt install -y wget software-properties-common
    DEBIAN_FRONTEND=noninteractive apt-get clean all

    if [ ! -f "/${CUSTOM_INSTALL}" ]; then
        # Add rocm repository
        #chmod 1777 /tmp
        #wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -;

        # Make the directory if it doesn't exist yet.
        # This location is recommended by the distribution maintainers.
        mkdir --parents --mode=0755 /etc/apt/keyrings

        # Download the key, convert the signing-key to a full
        # keyring required by apt and store in the keyring directory
        wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
            gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] $AMDGPU_DEB_REPO/ubuntu $ROCM_BUILD_NAME $ROCM_BUILD_NUM" | tee --append /etc/apt/sources.list.d/amdgpu.list
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" | tee /etc/apt/sources.list.d/rocm.list
        echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
             | tee /etc/apt/preferences.d/rocm-pin-600
    else
        bash "/${CUSTOM_INSTALL}"
    fi
    apt-get update --allow-insecure-repositories

    wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    echo "deb [arch=amd64 trusted=yes] http://apt.llvm.org/$DISTRO/ llvm-toolchain-$DISTRO-18 main" | tee /etc/apt/sources.list.d/llvm.list
    apt-get update --allow-insecure-repositories

    # install rocm
    /setup.packages.sh /devel.packages.rocm.txt

    MIOPENKERNELS=$( \
                        apt-cache search --names-only miopen-hip-gfx | \
                        awk '{print $1}' | \
                        grep -F -v . || \
		        true )
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ${MIOPENKERNELS}

    #install hipblasLT if available
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated hipblaslt-dev || true

elif [[ "$DISTRO" == "el7" ]]; then
    if [ ! -f "/${CUSTOM_INSTALL}" ]; then
        RPM_ROCM_REPO=http://repo.radeon.com/rocm/yum/${ROCM_VERS}/main
        echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/rocm.repo
        echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/${ROCM_VERS}/rhel/7/main/x86_64/\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/amdgpu.repo
    else
        bash "/${CUSTOM_INSTALL}"
    fi
    yum clean all

    # install rocm
    /setup.packages.rocm.cs7.sh /devel.packages.rocm.cs7.txt

    # install hipblasLT if available
    yum --enablerepo=extras install -y hipblaslt-devel || true

elif [[ "$DISTRO" == "el8" ]]; then
    if [ ! -f "/${CUSTOM_INSTALL}" ]; then
        RPM_ROCM_REPO=http://repo.radeon.com/rocm/rhel8/${ROCM_VERS}/main
        echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >>/etc/yum.repos.d/rocm.repo
        echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/${ROCM_VERS}/rhel/8.8/main/x86_64/\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >>/etc/yum.repos.d/amdgpu.repo
    else
        bash "/${CUSTOM_INSTALL}"
    fi
    dnf clean all

    # install rocm
    /setup.packages.rocm.el8.sh /devel.packages.rocm.el8.txt

    # install hipblasLT if available
    dnf --enablerepo=extras,epel,elrepo,build_system install -y hipblaslt-devel || true
fi

function ver { printf "%03d%03d%03d" $(echo "$1" | tr '.' ' '); }
# If hipcc uses llvm-17, in case of ROCM 6.0.x and 6.1.x and
# host compiler is llvm-18 leads to mismatch in name mangling resulting
# in faliure to link compiled gpu kernels. This linker option circumvents that issue.
if [ $(ver "$ROCM_VERSION") -lt $(ver "6.2.0") ]
then
  echo "build:rocm_base --copt=-fclang-abi-compat=17" >> /etc/bazel.bazelrc
fi

GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS:-"gfx908 gfx90a gfx940 gfx941 gfx942 gfx950 gfx1030 gfx1100"}

echo $ROCM_VERSION
echo $ROCM_REPO
echo $ROCM_PATH
echo $GPU_DEVICE_TARGETS

# Ensure the ROCm target list is set up
printf '%s\n' ${GPU_DEVICE_TARGETS} | tee -a "$ROCM_PATH/bin/target.lst"
touch "${ROCM_PATH}/.info/version"
