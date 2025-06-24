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

load("//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("//tensorflow/compiler/mlir/lite:symlink_files.bzl", "symlink_files")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/lite:__subpackages__"],
    features = [
        # Cannot use header_modules (parse_headers feature fails).
        "-use_header_modules",
    ],
    licenses = ["notice"],
)

# These flags are needed for parse_headers feature.
COPTS = [
    "-fexceptions",
    "-frtti",
]

pybind_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    copts = COPTS,
    pytype_srcs = [
        ":_mlirPyi",
    ],
    deps = [
        "@llvm-project//mlir:MLIRBindingsPythonCore",
        "@llvm-project//mlir:MLIRBindingsPythonNanobindHeaders",
        "@nanobind",
    ],
)

symlink_files(
    name = "_mlirPyi",
    srcs = [
        "@llvm-project//mlir/python:IRPyIFiles",
        "@llvm-project//mlir/python:PassManagerPyIFiles",
    ],
    dst = "_mlir",
    flatten = True,
)
