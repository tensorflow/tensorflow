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
#   Contains the Keras initializer API (internal TensorFlow version).

load("@xla//third_party/rules_python/python:defs.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = [
        "//tensorflow/python/keras/google/private_tf_api_test:__pkg__",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "__init__.py",
        "initializers_v1.py",
        "initializers_v2.py",
    ],
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_inspect",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/util:tf_export",
    ],
)
