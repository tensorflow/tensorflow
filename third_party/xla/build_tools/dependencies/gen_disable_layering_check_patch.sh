#!/bin/bash
# Copyright 2025 The OpenXLA Authors.
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


# Generates a patch file that disables the layering check for all cc_library
# targets in the archive. Both BUILD and BUILD.bazel files are taken into account.
#
# The script takes one argument: the URL of the .tar.gz archive to download.
#
# The following tools are needed (need to be installed on the machine):
# - curl
# - git
# - buildozer (from Bazel buildtools)
#
# The tool has originally been written for ortools but should work for similarly structured
# projects as well.
#
# Example:
# build_tools/dependencies/gen_disable_layering_check_patch.sh \
# https://github.com/google/or-tools/archive/v9.11.tar.gz \
# > third_party/ortools/layering_check.patch

set -euo pipefail

readonly TMP_DIR=$(mktemp -d)
trap 'rm -rf -- $TMP_DIR' EXIT

echo "Downloading archive $1..." >&2
curl -Lqo "$TMP_DIR/archive.tar.gz" "$1" 1>&2

echo "Extracting archive..." >&2
mkdir -p "$TMP_DIR/extracted" 1>&2
tar  -x -C "$TMP_DIR/extracted" -f "$TMP_DIR/archive.tar.gz" --strip-components=1 1>&2

echo "Initialzing temporary git repo..." >&2
git -C "$TMP_DIR/extracted" init 1>&2
git -C "$TMP_DIR/extracted" add . 1>&2
git -C "$TMP_DIR/extracted" commit --no-verify -m "original state" -q 1>&2

echo "Patching build targets..." >&2
find $TMP_DIR/extracted -name BUILD.bazel -or -name BUILD | while read f; do
   buildozer 'add features "-layering_check"' $(dirname $f):%cc_library 1>&2 || exit_code=$?
   if [[ $exit_code -ne 0 && $exit_code -ne 3 ]]; then
     echo "Buildozer command failed with exit code: $exit_code" >&2
     exit $exit_code
   fi
done

echo "Generating diff..." >&2
git -C "$TMP_DIR/extracted" --no-pager diff
