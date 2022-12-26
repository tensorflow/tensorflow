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

for wheel in /tf/pkg/*.whl; do
  echo "Checking and renaming $wheel..."
  # Change linux to manylinux
  NEW_TF_WHEEL=${wheel/linux/"manylinux2014"}

<<<<<<< HEAD:tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rename_and_verify_ROCM_wheels.sh
  mv $wheel $NEW_TF_WHEEL
  auditwheel show $NEW_TF_WHEEL
done
=======
on:
  workflow_dispatch:  # Allow manual triggers
  schedule:
    - cron: 0 4 * * *  # 4am UTC is 9pm PDT and 8pm PST
name: Set nightly branch to master HEAD
jobs:
  master-to-nightly:
    if: github.repository == 'tensorflow/tensorflow' # Don't do this in forks
    runs-on: ubuntu-latest
    steps:
    - uses: zofrex/mirror-branch@a8809f0b42f9dfe9b2c5c2162a46327c23d15266 # v1.0.3
      name: Set nightly branch to master HEAD
      with:
        target-branch: 'nightly'
>>>>>>> google_upstream/master:.github/workflows/update-nightly.yml
