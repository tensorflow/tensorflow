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
# Helper for retrying transient Bazel build/link and infra failures against a
# warm cache.

tfrun_bazel_with_retries() {
  local retries="${TFCI_BAZEL_COMMAND_RETRIES:-0}"
  if [[ ! "$retries" =~ ^[0-9]+$ ]]; then
    echo "==TFCI==: Invalid TFCI_BAZEL_COMMAND_RETRIES='${retries}'; expected a non-negative integer." >&2
    return 2
  fi

  local max_attempts=$((10#$retries + 1))
  local attempt
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    local rc=0
    if tfrun bazel "$@"; then
      return 0
    else
      rc=$?
    fi

    if (( attempt >= max_attempts )); then
      return "$rc"
    fi
    # Fail fast on 2 (CLI usage error), 3 (tests failed), 4 (no tests found),
    # and 8 (interrupted); they are never transient. Every other non-zero code
    # (build/link 1, infra 32-39, docker exec) is retried.
    if (( rc == 2 || rc == 3 || rc == 4 || rc == 8 )); then
      echo "==TFCI==: Bazel command failed with non-retryable exit code ${rc}; not retrying." >&2
      return "$rc"
    fi
    echo "==TFCI==: Bazel command failed with exit code ${rc} (attempt ${attempt}/${max_attempts}); retrying against warm Bazel cache..." >&2
  done
}
