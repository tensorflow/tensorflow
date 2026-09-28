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

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

pybind_extension(
    name = "register_list_ops_py",
    srcs = ["register_list_ops_py.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "register_list_ops_py.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/kernels/variants:list_ops_lib",
        "@pybind11",
    ],
)

py_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.py"],
    strict_deps = True,
    tags = ["nochromiumos_arm"],
    deps = [
        ":register_list_ops_py",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:test",
    ],
)
