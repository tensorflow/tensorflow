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
#
# ==============================================================================
set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_BUILD__JOBS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION

if [ ! -d /tf ];then
    # The bazelrc files expect /tf to exist
    mkdir /tf
fi

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/cpu.bazelrc test \
          --config=sigbuild_local_cache \
          --verbose_failures \
          --config=pycpp \
          --action_env=TF_NEED_ROCM=1 \
          --action_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
          --local_test_jobs=${N_BUILD_JOBS} \
	  --repo_env=ROCM_PATH=/opt/rocm \
          --jobs=${N_BUILD_JOBS}
