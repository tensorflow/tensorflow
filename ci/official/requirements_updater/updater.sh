#!/usr/bin/env bash
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

# script to run pip-compile for each requirement.
# if there is a change in requirements.in then all lock files will be updated
# accordingly

# All commands run relative to this directory
cd "$(dirname "${BASH_SOURCE[0]}")"

mv BUILD.bazel BUILD

SUPPORTED_VERSIONS=("3_9" "3_10" "3_11" "3_12")

for VERSION in "${SUPPORTED_VERSIONS[@]}"
do
  touch "requirements_lock_$VERSION.txt"
  bazel run \
    --experimental_convenience_symlinks=ignore \
    --enable_bzlmod=false \
    //:requirements_"$VERSION".update
  sed -i '/^#/d' requirements_lock_"$VERSION".txt
  mv requirements_lock_"$VERSION".txt ../../../requirements_lock_"$VERSION".txt
done

mv BUILD BUILD.bazel
