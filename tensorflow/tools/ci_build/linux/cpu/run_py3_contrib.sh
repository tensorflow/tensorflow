#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Run configure.
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export CC_OPT_FLAGS='-mavx'
export PYTHON_BIN_PATH=`which python3`
yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --test_tag_filters=-no_oss,-oss_excluded,-oss_serial,-gpu,-tpu,-benchmark-test -k \
    --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 --config=opt \
    --test_size_filters=small,medium --test_output=errors -- \
    //tensorflow/contrib/...
