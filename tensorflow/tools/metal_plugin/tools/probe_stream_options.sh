#!/usr/bin/env bash
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
