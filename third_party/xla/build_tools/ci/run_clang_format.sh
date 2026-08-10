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

source "$(dirname "${BASH_SOURCE[0]}")/shell_common.sh"

CLANG_FORMAT_VERSION="${CLANG_FORMAT_VERSION:-17.0.6}"
CLANG_FORMAT_PKG="clang-format==${CLANG_FORMAT_VERSION}"

BUILD_WORKSPACE_DIRECTORY="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
cd "$BUILD_WORKSPACE_DIRECTORY"

if ! command -v uv >/dev/null 2>&1; then
  echoerr "uv is required to run the clang-format check."
  echoerr "Install uv from https://docs.astral.sh/uv/"
  exit 1
fi

FIX=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX=true
      shift
      ;;
    *)
      set +x
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--fix]" >&2
      exit 1
      ;;
  esac
done

get_merge_base() {
  local REFERENCE
  local REMOTE="${REMOTE:-$(git remote -v | awk '/openxla\/xla/ { print $1; exit }')}"

  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echoerr "This script must be run inside a Git repository."
    exit 1
  fi

  if [ -n "${TARGET_REF}" ]; then
    REFERENCE="$TARGET_REF"
  else
    if [ -z "$REMOTE" ]; then
      echoerr "Could not find a git remote pointing to openxla/xla. Please add it as a remote."
      echoerr "Example: git remote add upstream https://github.com/openxla/xla.git"
      exit 1
    fi
    REFERENCE="${REMOTE}/main"
  fi

  MERGE_BASE=$(git merge-base "$REFERENCE" HEAD || true)
  if [ -z "$MERGE_BASE" ]; then
    echoerr "Could not find a common ancestor with $REFERENCE. Please fetch and rebase on main."
    echoerr "Example: git fetch ${REMOTE:-origin} main && git rebase ${REMOTE:-origin}/main"
    exit 1
  fi
}

get_merge_base

EXTRA_ARGS=(-p1 -style=file)
if [ "$FIX" = true ]; then
  EXTRA_ARGS+=(-i)
fi

# Run diff against the merge base.
# -U0: Context of 0 lines (ignore surrounding code)
# clang-format-diff: Checks only lines present in the diff
DIFF=$(git diff -U0 --no-color "$MERGE_BASE" -- '*.cc' '*.h' |
  uvx --from "$CLANG_FORMAT_PKG" clang-format-diff.py "${EXTRA_ARGS[@]}")

if [ -n "$DIFF" ]; then
  echoerr "Clang-format failed on the following changes:"
  echo "$DIFF"
  exit 1
fi
