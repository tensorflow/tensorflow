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
# Prints "yes" when the installed TensorFlow headers carry SP_StreamOptions,
# which the StreamExecutor C API grew after the last release. Lets one source
# tree build both in-tree and against an older installed TensorFlow.
set -u
include="$1"
tmp=$(mktemp -t sp_stream_options).cc
cat > "$tmp" <<'PROBE'
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
SP_StreamOptions probe;
PROBE
if "${CXX:-c++}" -x c++ -std=c++17 -fsyntax-only -I"$include" "$tmp" 2>/dev/null; then
  echo yes
else
  echo no
fi
rm -f "$tmp"
