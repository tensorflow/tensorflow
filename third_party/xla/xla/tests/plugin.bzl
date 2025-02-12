# Copyright 2017 The OpenXLA Authors.
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

# Example:
#
# plugins = {
#   "foo": {
#     "deps": [
#       "//tensorflow/compiler/plugin/foo:foo_lib",
#       "//tensorflow/compiler/plugin/foo:test_macros",
#     ],
#     "disabled_manifest": "tensorflow/compiler/plugin/foo/disabled_test_manifest.txt",
#     "copts": [],
#     "tags": [],
#     "args": []
#     "data": [
#       "//tensorflow/compiler/plugin/foo:disabled_test_manifest.txt",
#     ],
#   },
# }

plugins = {}
