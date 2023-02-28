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
# Check and rename wheels manually since auditwheel repair currently isn't aware of
# ROCM and will stip references from the binaries.
# Only rename wheels that need it.
set -euxo pipefail

for wheel in /tf/pkg/*.whl; do
  echo "Checking and renaming $wheel..."
  if [[ "${wheel}" != *"manylinux"* ]]; then
      NEW_TF_WHEEL=${wheel/linux/"manylinux2014"}
      echo "Rename ${wheel} to ${NEW_TF_WHEEL}"
      mv $wheel $NEW_TF_WHEEL
  else
      echo "${wheel} already renamed"
      NEW_TF_WHEEL=${wheel}
  fi

  # Verify the wheel, but only if it matches the current verison of python
  PYTRIM=$( \
          python3 --version | \
          awk '{print$2}' | \
          cut -d "." -f1,2 | \
	  tr -d ".")
  if [[ "${NEW_TF_WHEEL}" == *"cp${PYTRIM}"* ]]; then
      CURRENT_PY_VERS=`python3 --version`
      echo "Verifying $NEW_TF_WHEEL for ${CURRENT_PY_VERS%.*} ..."
      TF_WHEEL="$NEW_TF_WHEEL" bats /usertools/wheel_verification.bats --timing
  fi
done
