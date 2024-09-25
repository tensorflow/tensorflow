#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
if [ -z ${SYCL_TOOLKIT_PATH+x} ];
then
workspace=$1
action=$2
echo "Install Intel OneAPI in $workspace/oneapi"
cd $workspace
mkdir -p oneapi
if ! [ -f $workspace/l_BaseKit_p_2024.1.0.596.sh ]; then
  echo "Download oneAPI package"
  wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/fdc7a2bc-b7a8-47eb-8876-de6201297144/l_BaseKit_p_2024.1.0.596.sh
fi
bash l_BaseKit_p_2024.1.0.596.sh -a -s --eula accept --action $action --install-dir $workspace/oneapi --log-dir $workspace/oneapi/log --download-cache $workspace/oneapi/cache --components=intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.mkl.devel
else
  echo "SYCL_TOOLKIT_PATH set to $SYCL_TOOLKIT_PATH", skip install/remove oneAPI;
fi
