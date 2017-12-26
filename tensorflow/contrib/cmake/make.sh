#!/bin/sh

(
cd "$(dirname "$0")"
mkdir -p _build

(
cd _build
rm -rf -- *
cmake ..
)
)
