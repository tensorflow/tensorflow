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

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

py_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    strict_deps = True,
    deps = [
        ":authoring",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
    ],
)
