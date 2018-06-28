#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

OPENBLAS_SRC_PATH=/tmp/openblas_src/
POWER="POWER8"
USE_OPENMP="USE_OPENMP=1"
OPENBLAS_INSTALL_PATH="/usr"
apt-get update
apt-get install -y gfortran gfortran-5
sudo rm -rf ${OPENBLAS_SRC_PATH}
git clone -b release-0.3.0 https://github.com/xianyi/OpenBLAS ${OPENBLAS_SRC_PATH}
cd ${OPENBLAS_SRC_PATH}
# Pick up fix for OpenBLAS issue 1571
git cherry-pick -X theirs 961d25e9c7e4a1758adb1dbeaa15187de69dd052
make TARGET=${POWER} ${USE_OPENMP} FC=gfortran
make PREFIX=${OPENBLAS_INSTALL_PATH} install
