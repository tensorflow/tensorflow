#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# please run this at root directory of tensorflow
success=1

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

# only tensorflow/core/ops is checked at the moment for experimental purpose
for filename in $(find tensorflow/core/ops -name *.h -o -name *.cc); do
  $CLANG_FORMAT --style=google $filename | diff $filename - > /dev/null
  if [ ! $? -eq 0 ]; then
    success=0
    echo File $filename is not properly formatted with "clang-format --style=google"
  fi
done

if [ $success == 0 ]; then
  echo Clang format check fails.
  exit 1
fi

echo Clang format check success.
