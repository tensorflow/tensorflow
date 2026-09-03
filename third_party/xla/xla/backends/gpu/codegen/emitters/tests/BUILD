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
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

_FILECHECK_ONLY_TESTS = [
    "reduce_column_small/f32_32_v2.hlo",
    "reduce_column_small/s8_f32_32_v4.hlo",
    "scatter/sorted_indices_large.hlo",
]

lit_test_suite(
    name = "tests",
    srcs = glob(
        ["**/*.hlo"],
        exclude = _FILECHECK_ONLY_TESTS,
    ),
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_cuda_tests_tags(),
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
    tools = [
        "//xla/backends/gpu/codegen/tools:fusion_to_mlir",
        "//xla/backends/gpu/codegen/tools:gpu_test_correctness",
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

lit_test_suite(
    name = "filecheck_only_tests",
    srcs = _FILECHECK_ONLY_TESTS,
    cfg = "//xla:lit.cfg.py",
    default_tags = ["gpu"],
    tools = [
        "//xla/backends/gpu/codegen/tools:fusion_to_mlir",
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
