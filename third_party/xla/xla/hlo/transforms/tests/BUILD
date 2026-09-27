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

load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            # go/keep-sorted start
            "algebraic_simplifier.hlo",
            "cholesky_expander.hlo",
            "operand_upcaster.hlo",
            "optimization_barrier_expander.hlo",
            "rewrite_bf16_conv_to_onednn.hlo",
            "rng_bit_generator_expander.hlo",
            "rng_expander.hlo",
            # go/keep-sorted end
        ],
        include = [
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/hlo/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
