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

load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "mlir_wrapper",
    srcs = [
        "attrs.cc",
        "basic_classes.cc",
        "builders.cc",
        "mlir_wrapper.cc",
        "mlir_wrapper.h",
        "ops.cc",
        "types.cc",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "mlir_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "filecheck_wrapper",
    srcs = ["filecheck_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "filecheck_wrapper.pyi",
    ],
    starlark_only = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@pybind11",
    ],
)
