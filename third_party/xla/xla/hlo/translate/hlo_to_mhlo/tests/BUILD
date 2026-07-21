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
    name = "all_tests",
    srcs = enforce_glob(
        [
            "attributes.hlo",
            "auto_layout.hlo",
            "bool_compare.hlo",
            "call.hlo",
            "case_conditional.hlo",
            "composite_call.hlo",
            "custom_call.hlo",
            "dynamic_param.hlo",
            "frontend_attributes.hlo",
            "fully_connected_reference_model.hlo",
            "fusion.hlo",
            "if_conditional.hlo",
            "import.hlo",
            "import_locs.hlo",
            "import_async.hlo",
            "import_bounded_dynamism.hlo",
            "import_dynamic_reshape.hlo",
            "import_entry_computation_layout.hlo",
            "import_emit_stablehlo.hlo",
            "layouts_and_names.hlo",
            "location.hlo",
            "memory_space.hlo",
            "module_attributes.hlo",
            "module_config.hlo",
            "operand_convert_for_convolution.hlo",
            "ragged_dot.hlo",
            "simple.hlo",
            "spmd_module_sharding.hlo",
            "stablehlo.hlo",
            "stacktrace_to_location.hlo",
            "types.hlo",
            "while.hlo",
            "result_accuracy.hlo",
        ],
        include = [
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/hlo/tools:hlo-translate",
        "//xla/hlo/translate:xla-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
