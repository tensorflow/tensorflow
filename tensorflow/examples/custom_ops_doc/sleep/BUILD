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

# (non-blocking) op using AsyncOpKernel

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "sleep_kernel.so",
    srcs = [
        "sleep_kernel.cc",
        "sleep_op.cc",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

py_strict_library(
    name = "sleep_op",
    srcs = ["sleep_op.py"],
    data = ["sleep_kernel.so"],
    srcs_version = "PY3",
)

py_strict_binary(
    name = "sleep_bin",
    srcs = ["sleep_bin.py"],
    srcs_version = "PY3",
    deps = [
        ":sleep_op",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

tf_py_test(
    name = "sleep_test",
    size = "medium",  # This test blocks using sleep,
    timeout = "short",  # but it still runs quickly.
    srcs = ["sleep_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":sleep_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
