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
#   Contains User Ops (internal TensorFlow version).

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")

visibility = [
    "//tensorflow:internal",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
    out = "ops/gen_user_ops.py",
    py_lib_rule = py_library,
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops.py"],
    strict_deps = True,
    deps = [
        ":user_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)
