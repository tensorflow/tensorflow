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

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# Bundle together all of the test utilities that are used by tests.
exports_files(["run_lit.sh"])

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/core/ir/importexport:tfg-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = [
        "mlir",
        "pbtxt",
    ],
)
