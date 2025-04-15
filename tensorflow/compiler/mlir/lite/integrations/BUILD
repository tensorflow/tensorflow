# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/lite/integrations:__subpackages__",
        "//third_party/odml/litert/litert/python/tools/model_utils:__subpackages__",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "model_utils_core_pybind",
    srcs = [
        "model_utils_core_pybind.cc",
    ],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/lite/stablehlo:prepare_hlo",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/python/lib/core:ndarray_tensor",
        "//tensorflow/python/lib/core:py_func_lib",
        "//third_party/python_runtime:headers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@pybind11",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)

py_test(
    name = "py_bindings_test",
    srcs = ["py_bindings_test.py"],
    deps = [
        "//tensorflow/compiler/mlir/lite/integrations/python/mlir",
    ],
)
