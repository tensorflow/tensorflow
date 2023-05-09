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
# Usage: setup.rocm.sh <ROCM_VERSION>
set -x

# # Add the ROCm package repo location
ROCM_VERSION=$1 # e.g. 5.2.0
ROCM_PATH=${ROCM_PATH:-/opt/rocm-${ROCM_VERSION}}

if [ ! -f "/${CUSTOM_INSTALL}" ]; then
ROCM_VERSION_REPO=$(echo $ROCM_VERSION | grep -o "\w.\w") # e.g 5.2
RPM_ROCM_REPO=http://repo.radeon.com/rocm/yum/$(echo $ROCM_VERSION | grep -o "\w.\w")/main
echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/rocm.repo
echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/latest/rhel/7.9/main/x86_64/\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/amdgpu.repo
else
    bash "/${CUSTOM_INSTALL}"
fi

GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS:-"gfx900 gfx906 gfx908 gfx90a gfx1030"}

echo $ROCM_VERSION
echo $ROCM_REPO
echo $ROCM_PATH
echo $GPU_DEVICE_TARGETS

# install rocm
/setup.packages.rocm.cs7.sh /devel.packages.rocm.cs7.txt

# install hipblasLT if available
yum --enablerepo=extras install -y hipblaslt-devel || true

# Ensure the ROCm target list is set up
printf '%s\n' ${GPU_DEVICE_TARGETS} | tee -a "$ROCM_PATH/bin/target.lst"
touch "${ROCM_PATH}/.info/version"
