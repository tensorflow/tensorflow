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

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
)

licenses(["notice"])

py_test(
    name = "side_inputs_test",
    srcs = ["side_inputs_test.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "side_inputs_manual_api_test",
    srcs = ["side_inputs_manual_api_test.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)
