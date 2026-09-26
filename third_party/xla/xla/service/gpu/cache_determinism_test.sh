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
# ==============================================================================
set -e

# Setup paths
RUN_HLO_MODULE="${TEST_SRCDIR}/${TEST_WORKSPACE}/third_party/tensorflow/compiler/xla/tools/run_hlo_module"
HLO_FILE="${TEST_SRCDIR}/${TEST_WORKSPACE}/third_party/tensorflow/compiler/xla/tests/data/cudnn_reproducer.hlo"

OUT="${TEST_TMPDIR}/out"
mkdir -p "$OUT"

echo "Running first compilation..."
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_experimental_dump_gpu_executable=true --xla_dump_to=${OUT}" \
  "$RUN_HLO_MODULE" --platform=gpu "$HLO_FILE"

FILE1="${TEST_TMPDIR}/exec1.riegeli"
cp "${OUT}/"*.gpu_executable.riegeli "$FILE1"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "Running second compilation..."
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_experimental_dump_gpu_executable=true --xla_dump_to=${OUT}" \
  "$RUN_HLO_MODULE" --platform=gpu "$HLO_FILE"

FILE2="${TEST_TMPDIR}/exec2.riegeli"
cp "${OUT}/"*.gpu_executable.riegeli "$FILE2"

echo "Diffing outputs..."
if cmp "$FILE1" "$FILE2"; then
  echo "SUCCESS: Compilation cache outputs are deterministic."
  exit 0
else
  echo "FAIL: Compilation cache outputs are non-deterministic."
  exit 1
fi
