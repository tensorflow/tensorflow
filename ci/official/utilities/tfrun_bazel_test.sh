#!/bin/bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
#
# Unit tests for tfrun_bazel.sh.
# Run manually from the TensorFlow repository root:
#   bash ci/official/utilities/tfrun_bazel_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tfrun_bazel.sh"

TEST_TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TEST_TMPDIR}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# Stub tfrun that records calls and replays exit codes from MOCK_EXIT_CODES.
tfrun() {
  if [[ "${1:-}" != "bazel" ]]; then
    fail "Expected first arg 'bazel', got '${1:-}'"
  fi
  local count=0
  if [[ -f "${TEST_TMPDIR}/calls" ]]; then
    count="$(wc -l < "${TEST_TMPDIR}/calls")"
  fi
  echo "$*" >> "${TEST_TMPDIR}/calls"
  local idx=$((count + 1))
  local rc
  rc="$(awk -v n="${idx}" '{print $n}' <<< "${MOCK_EXIT_CODES}")"
  return "${rc:-0}"
}

reset_calls() {
  rm -f "${TEST_TMPDIR}/calls"
}

call_count() {
  if [[ ! -f "${TEST_TMPDIR}/calls" ]]; then
    echo 0
  else
    wc -l < "${TEST_TMPDIR}/calls" | tr -d ' '
  fi
}

# 1. Default 0 retries fails fast on exit code 1.
reset_calls
MOCK_EXIT_CODES="1 0"
unset TFCI_BAZEL_COMMAND_RETRIES
rc=0
tfrun_bazel_with_retries build //tensorflow/core/... || rc=$?
[[ "$rc" -eq 1 ]] || fail "Expected rc=1 with 0 retries, got ${rc}"
[[ "$(call_count)" -eq 1 ]] || fail "Expected 1 call, got $(call_count)"

# 2. 2 retries recover on attempt 2 and preserve caller errexit.
reset_calls
MOCK_EXIT_CODES="1 0"
TFCI_BAZEL_COMMAND_RETRIES=2
tfrun_bazel_with_retries test //tensorflow/core/... 2>"${TEST_TMPDIR}/stderr"
[[ "$(call_count)" -eq 2 ]] || fail "Expected 2 calls, got $(call_count)"
[[ "$-" == *e* ]] || fail "Expected errexit (-e) to remain enabled"
grep -q 'exit code 1 (attempt 1/3)' "${TEST_TMPDIR}/stderr" || fail "Expected retry diagnostic on stderr"

# 3. Exit code 3 (tests failed) after a retried build failure fails fast.
reset_calls
MOCK_EXIT_CODES="1 3 0"
TFCI_BAZEL_COMMAND_RETRIES=2
rc=0
tfrun_bazel_with_retries test //tensorflow/core/... 2>"${TEST_TMPDIR}/stderr" || rc=$?
[[ "$rc" -eq 3 ]] || fail "Expected rc=3 on test failure, got ${rc}"
[[ "$(call_count)" -eq 2 ]] || fail "Expected 2 calls, got $(call_count)"

# 4. Leading-zero "08" parses as base-10 8 retries; recovers on attempt 3.
reset_calls
MOCK_EXIT_CODES="1 36 0"
TFCI_BAZEL_COMMAND_RETRIES="08"
tfrun_bazel_with_retries build //tensorflow/core/... 2>"${TEST_TMPDIR}/stderr"
[[ "$(call_count)" -eq 3 ]] || fail "Expected 3 calls, got $(call_count)"

# 5. Exhausted retries return the final exit code.
reset_calls
MOCK_EXIT_CODES="1 1 36"
TFCI_BAZEL_COMMAND_RETRIES=2
rc=0
tfrun_bazel_with_retries build //tensorflow/tools/pip_package:wheel 2>"${TEST_TMPDIR}/stderr" || rc=$?
[[ "$rc" -eq 36 ]] || fail "Expected rc=36 after exhausted retries, got ${rc}"
[[ "$(call_count)" -eq 3 ]] || fail "Expected 3 calls, got $(call_count)"

# 6. Exit code 2 (bad CLI usage), 4 (no tests found), and 8 (interrupted) fail fast.
for non_retryable_rc in 2 4 8; do
  reset_calls
  MOCK_EXIT_CODES="${non_retryable_rc} 0"
  TFCI_BAZEL_COMMAND_RETRIES=2
  rc=0
  tfrun_bazel_with_retries test //tensorflow/core/... 2>"${TEST_TMPDIR}/stderr" || rc=$?
  [[ "$rc" -eq "$non_retryable_rc" ]] || fail "Expected rc=${non_retryable_rc}, got ${rc}"
  [[ "$(call_count)" -eq 1 ]] || fail "Expected 1 call for rc=${non_retryable_rc}, got $(call_count)"
  grep -q "non-retryable exit code ${non_retryable_rc}" "${TEST_TMPDIR}/stderr" || fail "Expected fail-fast diagnostic on stderr"
done

# 7. Non-numeric TFCI_BAZEL_COMMAND_RETRIES fails fast with exit code 2.
reset_calls
MOCK_EXIT_CODES="0"
TFCI_BAZEL_COMMAND_RETRIES="invalid"
rc=0
tfrun_bazel_with_retries build //tensorflow/core/... 2>"${TEST_TMPDIR}/stderr" || rc=$?
[[ "$rc" -eq 2 ]] || fail "Expected rc=2 on invalid retries, got ${rc}"
[[ "$(call_count)" -eq 0 ]] || fail "Expected 0 calls on invalid retries, got $(call_count)"

echo "PASS: all tfrun_bazel.sh unit tests succeeded."
