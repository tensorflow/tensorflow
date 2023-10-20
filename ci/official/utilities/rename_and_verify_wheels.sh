#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# Check and rename wheels with auditwheel. Inserts the platform tags like
# "manylinux_xyz" into the wheel filename.
set -euxo pipefail

DIR=$1
find $DIR -iname "*.whl" | while read wheel; do
  echo "Checking and renaming $wheel..."
  wheel=$(realpath "$wheel")
  # Repair wheel based upon name/architecture, fallback to x86
  if [[ $wheel == *"aarch64.whl" ]]; then
    time python3 -m auditwheel repair --plat manylinux2014_aarch64 "$wheel" --wheel-dir build 2>&1 | tee check.txt
  else
    time python3 -m auditwheel repair --plat manylinux2014_x86_64 "$wheel" --wheel-dir build 2>&1 | tee check.txt
  fi

  # We don't need the original wheel if it was renamed
  new_wheel=$(awk '/Fixed-up wheel written to/ {print $NF}' check.txt)
  if [[ "$new_wheel" != "$wheel" ]]; then
    rm "$wheel"
    wheel="$new_wheel"
  fi
  rm check.txt

  TF_WHEEL="$wheel" bats ./ci/official/utilities/wheel_verification.bats --timing
done
