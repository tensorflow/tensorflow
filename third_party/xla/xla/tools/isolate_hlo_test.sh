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

ISOLATE_HLO="$1"
FILECHECK="$2"
TEST_HLO_FILE="$3"

OUTPUT_HLO_FILE="${TEST_TMPDIR}/extracted.hlo"

# Run isolate_hlo on the convolution instructions.
"$ISOLATE_HLO" \
  --input="$TEST_HLO_FILE" \
  --output="$OUTPUT_HLO_FILE" \
  --instruction_name="conv" \
  --input_format="txt" \
  --output_format="long_txt"

# Verify the extracted output contains the isolated instruction in standard format.
cat "$OUTPUT_HLO_FILE" | "$FILECHECK" -v -input-file /dev/stdin "$0"

echo "isolate_hlo test passed."
exit 0

# CHECK: HloModule conv
# CHECK: ENTRY %entry_computation
# CHECK: %input = f32[1,224,224,3]{{{[0-9,]*}}} parameter(0)
# CHECK: %filter = f32[7,7,3,64]{{{[0-9,]*}}} parameter(1)
# CHECK: ROOT %conv = f32[1,112,112,64]{{{[0-9,]*}}} convolution(%input, %filter), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
