#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"

echo "Building host tools now..."
mkdir -p host && cd host
cmake ${SCRIPT_DIR}/../make/downloads/flatbuffers
cmake --build .
cd ..

echo "Building target libraries and tools now..."
mkdir -p target && cd target
cmake "$@" ${SCRIPT_DIR}/..
cmake --build .
cd ..