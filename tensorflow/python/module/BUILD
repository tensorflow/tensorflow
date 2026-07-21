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
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "module",
    srcs = ["module.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":module",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute:ps_values",
        "//tensorflow/python/distribute:tpu_values",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
