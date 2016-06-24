#!/bin/bash -eux
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Bash test for assuring that the runner_flags are imported and learn_runner
# runs successfully.
#

# Helper functions
# Exit after a failure
die() {
  echo "$@"
  exit 1
}

DIR="$TEST_SRCDIR"

# Check if TEST_WORKSPACE is defined, and set as empty string if not.
if [ -z "${TEST_WORKSPACE-}" ]
then
  TEST_WORKSPACE=""
fi

if [ ! -z "$TEST_WORKSPACE" ]
then
  DIR="$DIR"/"$TEST_WORKSPACE"
fi

LEARN_DIR="${DIR}/tensorflow/contrib/learn"
RUNNER_BIN="${LEARN_DIR}/runner_flags_test_util"

${RUNNER_BIN} --output_dir="/tmp" || die "Test failed, default flag values."
${RUNNER_BIN} --output_dir="/tmp" \
    --schedule="local_test" || die "Test failed, local_test schedule."

echo "PASS"
