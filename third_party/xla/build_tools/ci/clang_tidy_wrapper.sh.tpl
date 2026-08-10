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
# ==============================================================================

set -e

echoerr() {
  RED='\033[1;31m'
  NOCOLOR='\033[0m'
  printf "${RED}ERROR:${NOCOLOR} %s\n" "$*" >&2
}

REAL_BIN="$PWD/external/%LLVM_REPO_NAME%/bin/clang-tidy"
if [ ! -f "$REAL_BIN" ]; then
  echoerr "Failed to locate clang-tidy binary at: $REAL_BIN"
  exit 1
fi
echo "Using clang-tidy at: " $REAL_BIN
REAL_LIB_DIR="$(dirname "$REAL_BIN")/../lib"
export LD_LIBRARY_PATH="${REAL_LIB_DIR}:${LD_LIBRARY_PATH}"
# Intentional unquoted $@ expansion.
# Word-splitting is required here to resolve composite tokens passed by Bazel
# (e.g., "-include file.h" -> "-include" "file.h").
exec "$REAL_BIN" $@
