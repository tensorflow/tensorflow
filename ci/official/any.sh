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
set -euxo pipefail
cd "$(dirname "$0")/../../"  # tensorflow/
if [[ -n "${TF_ANY_SCRIPT:-}" ]]; then
  cp "$TFCI" any
  echo "source ci/official/envs/disable_all_uploads" >> any
  export TFCI=$(realpath any)
  "$TF_ANY_SCRIPT"
else
  source "${BASH_SOURCE%/*}/utilities/setup.sh"
  read -ra TARGETS_AS_ARRAY <<<"$TF_ANY_TARGETS"
  tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" "${TF_ANY_MODE:-test}" "${TFCI_BAZEL_COMMON_ARGS[@]}" "${TARGETS_AS_ARRAY[@]}"
fi
