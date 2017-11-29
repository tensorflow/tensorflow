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
"""Additional XLA devices to be included in the unit test suite."""

# If you wish to edit this file without checking it into the repo, consider:
#   git update-index --assume-unchanged tensorflow/compiler/tests/plugin.bzl

plugins = {
  #"poplar": {"device":"XLA_IPU", "types":"DT_FLOAT,DT_INT32", "tags":[]},
}

