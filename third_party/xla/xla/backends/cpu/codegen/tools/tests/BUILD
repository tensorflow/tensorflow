# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "ir_compiler_opt_tests",
    srcs = [
        "exp_inlines.ll",
        "ir_compiler_basic_test.ll",
        "ir_compiler_vectorization_test.ll",
        "ldexp_inlines.ll",
    ],
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/backends/cpu/codegen/tools:ir-compiler-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
