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

load("//tensorflow:py.default.bzl", "py_library")
load("//tensorflow/compiler/mlir/lite:symlink_files.bzl", "symlink_inputs")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/lite:__subpackages__"],
    features = [
        # Cannot use header_modules (parse_headers feature fails).
        "-use_header_modules",
    ],
    licenses = ["notice"],
)

symlink_inputs(
    name = "mlir_libs",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "_mlir_libs/": ["@llvm-project//mlir/python:MlirLibsPyFiles"],
    }},
)

py_library(
    name = "mlir",
    deps = [
        ":mlir_libs",
        "//tensorflow/compiler/mlir/lite/integrations/python/mlir/_mlir_libs:_mlir",
    ],
)
