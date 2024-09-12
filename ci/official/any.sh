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
# any.sh has two run modes.
#
# 1. RUN, TEST, OR BUILD BAZEL TARGET(S) WITHIN A TFCI ENVIRONMENT
#    To use:
#       export TFCI=ci/official/envs/env_goes_here
#       export TF_ANY_TARGETS="quoted list of targets, like on the command line"
#       export TF_ANY_MODE="test" or "build" or "run" (default: "test")
#       ./any.sh
#
# 2. RUN ANY OTHER SCRIPT AND ENV WITH NO SIDE EFFECTS (NO UPLOADS)
#    To use:
#       export TFCI=ci/official/envs/env_goes_here
#       export TF_ANY_SCRIPT=ci/official/wheel.sh
#       ./any.sh
#
# 3. DO THE SAME WITH A LOCAL CACHE OR RBE:
#       export TF_ANY_EXTRA_ENV=ci/official/envs/public_cache,ci/official/envs/disk_cache
#       ...
#       ./any.sh
#     or
#       export TF_ANY_EXTRA_ENV=ci/official/envs/local_rbe
#       ./any.sh
#       ...
set -euxo pipefail
cd "$(dirname "$0")/../../"  # tensorflow/
# Any request that includes "nightly_upload" should just use the
# local multi-cache (public read-only cache + disk cache) instead.
export TFCI="$(echo $TFCI | sed 's/,nightly_upload/,public_cache,disk_cache/')"
if [[ -n "${TF_ANY_EXTRA_ENV:-}" ]]; then
  export TFCI="$TFCI,$TF_ANY_EXTRA_ENV"
fi
if [[ -n "${TF_ANY_SCRIPT:-}" ]]; then
  "$TF_ANY_SCRIPT"
elif [[ -n "${TF_ANY_TARGETS:-}" ]]; then
  source "${BASH_SOURCE%/*}/utilities/setup.sh"
  tfrun bazel "${TF_ANY_MODE:-test}" $TFCI_BAZEL_COMMON_ARGS $TF_ANY_TARGETS
else
  echo 'Looks like $TF_ANY_TARGETS are $TF_ANY_SCRIPT are both empty. That is an error.'
  exit 1
fi
