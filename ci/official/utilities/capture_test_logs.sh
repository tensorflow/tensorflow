#!/bin/bash
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
ROOT_DIR=$1
OUTPUT_DIR=$2
mkdir -p $OUTPUT_DIR
cd $ROOT_DIR
find -L bazel-testlogs -name "test.log" -exec cp --parents {} "$OUTPUT_DIR" \;
find -L bazel-testlogs -name "test.xml" -exec cp --parents {} "$OUTPUT_DIR" \;
find -L "$OUTPUT_DIR" -name "test.log" -exec chmod -x {} \;
find -L "$OUTPUT_DIR" -name "test.log" -execdir mv test.log sponge_log.log \;
find -L "$OUTPUT_DIR" -name "test.xml" -execdir mv test.xml sponge_log.xml \;
