#!/usr/bin/env bash
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
# The two symbols TensorFlow looks up by name must be exported, and nothing
# may be left undefined that the process will not already have.
set -euo pipefail
lib="$1"

# grep -c rather than grep -q, and the table read once rather than per symbol.
# `grep -q` looks tidier and is wrong under `set -o pipefail`: it exits at the
# first match, whatever is writing the other end of the pipe takes a SIGPIPE,
# and the pipeline reports a failure that reads here as the symbol being
# absent. It only shows up once the library is big enough that the writer is
# still going when grep leaves, which the in-tree build produces at ten
# megabytes and the out-of-tree one, at one, does not. grep -c reads its input
# to the end.
exported=$(nm -gU "$lib")

missing=0
for symbol in _SE_InitPlugin _TF_InitKernel; do
  found=$(printf '%s\n' "$exported" | grep -c " T ${symbol}\$" || true)
  if [ "$found" -eq 0 ]; then
    echo "not exported: ${symbol#_}"
    missing=1
  fi
done

# Undefined symbols are expected, but only ones TensorFlow or the system
# frameworks provide. Anything else means a source file was left out.
undefined=$(nm -u "$lib" | sed 's/^ *//' | grep -v '^$' || true)
unresolved=$(echo "$undefined" | grep -E '_ZN10tensorflow5metal' || true)
if [ -n "$unresolved" ]; then
  echo "unresolved symbols from the plugin's own namespace:"
  echo "$unresolved"
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  echo "symbol check failed"
  exit 1
fi
echo "symbol check passed: SE_InitPlugin and TF_InitKernel exported, no self-references left dangling"
