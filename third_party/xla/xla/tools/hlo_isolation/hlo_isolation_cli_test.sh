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

HLO_ISOLATION_CLI="$1"
FILECHECK="$2"
TEST_HLO_FILE="$3"

# First run: Standard run with custom error bounds to verify correct option configuration.
"$HLO_ISOLATION_CLI" \
  --hlo_file="$TEST_HLO_FILE" \
  --test_platform=cpu \
  --reference_platform="" \
  --abs_error_bound=0.05 \
  --rel_error_bound=0.2 | "$FILECHECK" "$0" --check-prefix=RUN1

# Second run: Run with a skipped module to verify the success counter.
"$HLO_ISOLATION_CLI" \
  --hlo_file="$TEST_HLO_FILE" \
  --test_platform=cpu \
  --reference_platform="" \
  --skip_by_name="fusion" | "$FILECHECK" "$0" --check-prefix=RUN2

echo "hlo_isolation_cli test passed."
exit 0


# RUN1: Submodule: fusion
# RUN1: Success: YES
# RUN1: Reason: STAGE_1_DEFUSED_TPU_SUCCESS
# RUN1: Total run: 1, Success: 1

# RUN2: Submodule: fusion
# RUN2: Success: NO
# RUN2: Reason: MATCH_SKIP_BY_NAME
# RUN2: Total run: 1, Success: 0
