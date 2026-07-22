# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Description:
# Comprehensive tests for the TensorFlow SavedModels.

load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "variable_wrapper_test",
    srcs = ["variable_wrapper_test.py"],
    deps = [
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
    ],
)
