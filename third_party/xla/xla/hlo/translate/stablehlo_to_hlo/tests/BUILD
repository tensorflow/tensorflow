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

load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "simple.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    data = [":test_utilities"],
    tools = [
        "//xla/hlo/translate:xla-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
)
