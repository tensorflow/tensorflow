#!/usr/bin/env bash
# The two symbols TensorFlow looks up by name must be exported, and nothing
# may be left undefined that the process will not already have.
set -euo pipefail
lib="$1"

missing=0
for symbol in _SE_InitPlugin _TF_InitKernel; do
  if ! nm -gU "$lib" | grep -q " T ${symbol}\$"; then
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
