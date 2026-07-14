#!/bin/bash
set -e

echoerr() {
  RED='\033[1;31m'
  NOCOLOR='\033[0m'
  printf "${RED}ERROR:${NOCOLOR} %s\n" "$*" >&2
}

REAL_BIN="$PWD/external/%LLVM_REPO_NAME%/bin/clang-apply-replacements"
if [ ! -f "$REAL_BIN" ]; then
  echoerr "Failed to locate clang-apply-replacements binary at: $REAL_BIN"
  exit 1
fi
echo "Using clang-apply-replacements at: " $REAL_BIN
REAL_LIB_DIR="$(dirname "$REAL_BIN")/../lib"
export LD_LIBRARY_PATH="${REAL_LIB_DIR}:${LD_LIBRARY_PATH}"
exec "$REAL_BIN" "$@"
