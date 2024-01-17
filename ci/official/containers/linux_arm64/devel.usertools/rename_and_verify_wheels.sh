#!/usr/bin/env bash
#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# Check and rename wheels with auditwheel. Inserts the platform tags like
# "manylinux_xyz" into the wheel filename.
set -euxo pipefail

for wheel in /tf/pkg/*.whl; do
  echo "Checking and renaming $wheel..."
  time python3 -m auditwheel repair --plat manylinux2014_aarch64 "$wheel" --wheel-dir /tf/pkg 2>&1 | tee check.txt

  # We don't need the original wheel if it was renamed
  new_wheel=$(grep --extended-regexp --only-matching '/tf/pkg/\S+.whl' check.txt)
  if [[ "$new_wheel" != "$wheel" ]]; then
    rm "$wheel"
    wheel="$new_wheel"
  fi

  TF_WHEEL="$wheel" bats /usertools/wheel_verification.bats --timing
done
