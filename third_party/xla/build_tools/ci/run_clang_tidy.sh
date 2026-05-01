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
set -xe

BUILD_WORKSPACE_DIRECTORY=${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}
cd "$BUILD_WORKSPACE_DIRECTORY"
BAZEL_CMD=${BAZEL_CMD:-bazelisk}

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Error: This script must be run inside a Git repository."
  exit 1
fi
TARGET_REF=${TARGET_REF:-origin/main}
MERGE_BASE=$(git merge-base "$TARGET_REF" HEAD || true)
if [ -z "$MERGE_BASE" ]; then
  echo "Could not find a common ancestor with $TARGET_REF. Please rebase on upstream main." >&2
  exit 1
fi
CHANGED_FILES=$(git diff --name-only "$MERGE_BASE" | grep -E '\.(cc|h)$' || true)
# Always exit with 0 if no C++ files are changed.
if [ -z "$CHANGED_FILES" ]; then
  echo "No C++ files changed."
  exit 0
fi
PACKAGES=$(echo "$CHANGED_FILES" | while read -r file; do
    echo "//$(dirname "$file"):all"
done | sort -u | tr '\n' ' ')
BASE_QUERY="kind('cc_(library|binary|test)', rdeps(set($PACKAGES), set($CHANGED_FILES), 1))"
if [ -n "$TAGS_TO_IGNORE" ]; then
  QUERY="(${BASE_QUERY} except attr('tags', '${TAGS_TO_IGNORE}', //xla/...))"
else
  QUERY="${BASE_QUERY}"
fi
TARGETS=$($BAZEL_CMD query --config=clang-tidy "$QUERY")
if [ -z "$TARGETS" ]; then
  echo "No relevant targets found for changed files."
  exit 0
fi
$BAZEL_CMD build --config=clang-tidy --keep_going $TARGETS
