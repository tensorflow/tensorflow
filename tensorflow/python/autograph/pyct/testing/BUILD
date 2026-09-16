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
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "codegen",
    srcs = [
        "codegen.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:templates",
        "//third_party/py/numpy",
        "@pypi//gast",
    ],
)

py_library(
    name = "decorators",
    srcs = ["decorators.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "basic_definitions",
    srcs = ["basic_definitions.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    strict_deps = True,
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/platform:client_testlib",
    ],
)
