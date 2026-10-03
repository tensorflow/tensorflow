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

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "tf_py_strict_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/quantization/stablehlo/python:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    tags = ["no_pip"],
    visibility = ["//visibility:public"],
)

tf_py_strict_test(
    name = "testing_test",
    srcs = ["testing_test.py"],
    deps = [
        ":testing",
        "//tensorflow/python/platform:client_testlib",
    ],
)
