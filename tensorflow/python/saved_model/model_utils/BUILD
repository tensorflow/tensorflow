# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#   Keras saving and loading libraries.

load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],
)

py_strict_library(
    name = "model_utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":export_output",
        ":export_utils",
        ":mode_keys",
    ],
)

py_strict_library(
    name = "export_output",
    srcs = ["export_output.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_strict_test(
    name = "export_output_test",
    srcs = ["export_output_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":export_output",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model:signature_constants",
    ],
)

py_strict_library(
    name = "export_utils",
    srcs = ["export_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":export_output",
        ":mode_keys",
        "//tensorflow/python:op_selector",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_test(
    name = "export_test",
    srcs = ["export_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":export_output",
        ":export_utils",
        ":mode_keys",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
    ],
)

py_strict_library(
    name = "mode_keys",
    srcs = ["mode_keys.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:compat"],
)

py_strict_test(
    name = "mode_keys_test",
    srcs = ["mode_keys_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mode_keys",
        "//tensorflow/python:client_testlib",
    ],
)
