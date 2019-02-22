# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#   Contains the Keras Mixed Precision API (TensorFlow version).

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "autocast_variable",
    srcs = [
        "autocast_variable.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/distribute:values",
    ],
)

py_test(
    name = "autocast_variable_test",
    size = "medium",
    srcs = ["autocast_variable_test.py"],
    deps = [
        ":autocast_variable",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)
