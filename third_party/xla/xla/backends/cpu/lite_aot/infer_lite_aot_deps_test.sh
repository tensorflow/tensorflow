#!/bin/bash
# Copyright 2026 The OpenXLA Authors.
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

set -e

TOOL=$1
AOT_FILE=$2
EXPECTED=$3
DEPS_FILE="${TEST_TMPDIR}/out.deps"

# Run the tool and write output to a file
$TOOL --compilation_result=$AOT_FILE --output_deps=$DEPS_FILE

# Read the content of the deps file
OUTPUT=$(cat "$DEPS_FILE")

# Check if output contains the expected string
if [[ "$OUTPUT" != *"$EXPECTED"* ]]; then
  echo "TEST FAILED: Expected output to contain '$EXPECTED', got '$OUTPUT'"
  exit 1
fi

echo "TEST PASSED"
exit 0
