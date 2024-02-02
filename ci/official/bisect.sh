#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# Run any CI script and env configuration to bisect a failing target in some
# build configuration. You must set the following variables to control this
# script:
#
#   TF_BISECT_GOOD: Last known good commit (e.g. commit from the last passing job)
#   TF_BISECT_BAD: First bad commit (e.g. commit from the first failing job)
#   TF_BISECT_SCRIPT: The build script path relative to the TF root dir, e.g.
#     ci/official/wheel.sh
#   TFCI: The env config path, relative to the TF root dir, e.g.
#     ci/official/envs/an_env_config
#
# Note that you can combine bisect.sh with any.sh to bisect a single test:
#
#   export TFCI=...
#   export TF_BISECT_SCRIPT=ci/official/any.sh
#   export TF_BISECT_GOOD=a_good_commit_sha
#   export TF_BISECT_BAD=a_failing_commit_sha
#   export TF_ANY_TARGETS="quoted list of targets, like on the command line"
#   export TF_ANY_MODE=test
set -euxo pipefail
cd "$(dirname "$0")/../../"  # tensorflow/
export TFCI="$(echo $TFCI | sed 's/,nightly_upload/,public_cache,disk_cache/')"
git bisect start "$TF_BISECT_BAD" "$TF_BISECT_GOOD"
git bisect run $TF_BISECT_SCRIPT
