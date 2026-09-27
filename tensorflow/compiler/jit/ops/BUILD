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
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "xla_ops_wrapper_py",
    out = "xla_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_library,
    deps = ["//tensorflow/compiler/jit/ops:xla_ops"],
)

py_library(
    name = "xla_ops_grad",
    srcs = ["xla_ops_grad.py"],
    strict_deps = True,
    visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/python/ops:__pkg__",
    ],
    deps = ["//tensorflow/python/framework:ops"],
)
