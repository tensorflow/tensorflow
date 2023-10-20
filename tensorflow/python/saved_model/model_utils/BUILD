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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:op_selector",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/platform:client_testlib",
    ],
)
