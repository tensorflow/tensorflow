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
    name = "directives",
    srcs = ["directives.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "special_functions",
    srcs = ["special_functions.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python/autograph/operators:data_structures",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_test(
    name = "special_functions_test",
    srcs = ["special_functions_test.py"],
    strict_deps = True,
    deps = [
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)
