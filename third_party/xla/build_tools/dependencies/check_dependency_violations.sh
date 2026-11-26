#!/bin/bash
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================
# Checks for Bazel dependency violations for a given forbidden tag
# (gpu, cuda-only, or rocm-only) across //xla/...
#
# Usage:
#   [BAZEL_CONFIG=<config>] ./build_tools/dependencies/check_dependency_violations.sh <tag>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: [BAZEL_CONFIG=<config>] $0 <gpu|cuda-only|rocm-only>" >&2
  exit 2
fi

readonly CHECK_TAG="$1"
readonly TAG_WITH_UNDERSCORES="${CHECK_TAG//-/_}"

BAZEL_ARGS=()
if [[ -n "${BAZEL_CONFIG:-}" ]]; then
  BAZEL_ARGS+=(--config="${BAZEL_CONFIG}")
fi

if ! OUTPUT=$(bazel cquery "${BAZEL_ARGS[@]}" \
    --aspects "build_tools/dependencies/aspects.bzl%validate_${TAG_WITH_UNDERSCORES}_tag" \
    //xla/... 2>&1); then
  echo "Failed to run bazel cquery. Output:"
  echo "$OUTPUT"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- :x: \`${CHECK_TAG}\`: \`bazel cquery\` failed" >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 1
fi

if [[ "$OUTPUT" == *Violation* ]]; then
  VIOLATIONS=$(echo "$OUTPUT" | grep 'Violation' | sed -e 's/^.*\[Violation\]/ -/')
  echo "The following dependency violations were found:"
  echo "$VIOLATIONS"
  echo ""
  echo ""
  echo "There are a couple of potential solutions for this/these violation(s):"
  echo ""
  echo "1. Tag the dependent target with the same tag as the dependee."
  echo ""
  echo "2. If unavoidable make the dependency selective using the"
  echo "   'if_{gpu|cuda|rocm}_is_configured' macro. This is discouraged"
  echo "   outside of stream_executor."
  echo ""
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "- :x: \`${CHECK_TAG}\`: dependency violations found"
      echo '```'
      echo "$VIOLATIONS"
      echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 1
fi

echo "No dependency violations found for tag '${CHECK_TAG}'."
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  echo "- :white_check_mark: \`${CHECK_TAG}\`: no dependency violations" >> "$GITHUB_STEP_SUMMARY"
fi
