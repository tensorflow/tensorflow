#!/bin/bash
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

# Stores shared and platform-specific benchmark configurations

# Path to store downloaded TensorFlow models.
export TF_GRAPHS=~/tf-graphs
export BUILDER=bazel

configure_build() {
  cd ../../..
  yes "" | ./configure
}

benchmark_command() {
  echo "--config=opt --dynamic_mode=off //tensorflow/tools/benchmark:benchmark_model -- "
}
