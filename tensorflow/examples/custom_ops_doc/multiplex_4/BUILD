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

# Build multiplex_4 custom ops examples, which is similar to np.where

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_4_kernel.so",
    srcs = [
        "multiplex_4_kernel.cc",
        "multiplex_4_op.cc",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

py_strict_library(
    name = "multiplex_4_op",
    srcs = ["multiplex_4_op.py"],
    data = ["multiplex_4_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_library(
    name = "model_using_multiplex",
    srcs = ["model_using_multiplex.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "multiplex_4_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    srcs = ["multiplex_4_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":model_using_multiplex",
        ":multiplex_4_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "multiplex_2_save",
    srcs = ["multiplex_2_save.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "//tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "multiplex_4_load_use",
    srcs = ["multiplex_4_load_use.py"],
    srcs_version = "PY3",
    deps = [
        "model_using_multiplex",
        "multiplex_4_op",
        "@absl_py//absl:app",
    ],
)
