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

# Build multiplex_3 custom ops example, which is similar to np.where.
# This example shows how a Python wrapper can choose either to use "dispach
# for custom object types" to choose an old C++ Op (that supports only dense
# tensors) for backwards compatibility or a new C++ for new functionality
# (that supports sparse tensors).

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_3_kernel.so",
    srcs = [
        "multiplex_3_kernel.cc",
        "multiplex_3_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_3_op",
    srcs = ["multiplex_3_op.py"],
    data = [":multiplex_3_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
    ],
)

tf_py_test(
    name = "multiplex_3_test",
    size = "small",
    srcs = ["multiplex_3_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_3_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
