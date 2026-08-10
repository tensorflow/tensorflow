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
#   Common functionality for TensorFlow tooling

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "VERSION_MAJOR", "py_test")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//third_party/py/tf_keras/api:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "public_api",
    srcs = ["public_api.py"],
    strict_deps = True,
    deps = ["//tensorflow/python/util:tf_inspect"],
)

py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    data = if_pywrap(
        if_false = select({
            "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.dylib"],
            "//tensorflow:windows": [],
            "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION_MAJOR],
        }),
    ),
    strict_deps = True,
    deps = [
        ":public_api",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "traverse",
    srcs = ["traverse.py"],
    strict_deps = True,
    deps = ["//tensorflow/python/util:tf_inspect"],
)

py_test(
    name = "traverse_test",
    srcs = ["traverse_test.py"],
    data = if_pywrap(
        if_false = select({
            "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.dylib"],
            "//tensorflow:windows": [],
            "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION_MAJOR],
        }),
    ),
    strict_deps = True,
    deps = [
        ":test_module1",
        ":test_module2",
        ":traverse",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "test_module1",
    srcs = ["test_module1.py"],
    strict_deps = True,
    deps = [
        ":test_module2",
    ],
)

py_library(
    name = "test_module2",
    srcs = ["test_module2.py"],
    strict_deps = True,
)
