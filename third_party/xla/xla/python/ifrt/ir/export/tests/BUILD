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
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/python/ifrt/ir:internal"]),
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    data = [
        "ifrt_legalize_to_vifrt.0_1_0.mlir.bytes",
        "ifrt_legalize_to_vifrt.0_2_0.mlir.bytes",
        "ifrt_legalize_to_vifrt.0_3_0.mlir.bytes",
        "ifrt_legalize_to_vifrt.0_4_0.mlir.bytes",
    ],
    tools = [
        ":ifrt-translate",
        "//xla/python/ifrt/ir/tests:ifrt-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "ifrt-translate",
    testonly = True,
    srcs = ["ifrt-translate.cc"],
    deps = [
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:ifrt_ir_program_serdes",  # build_cleaner: keep
        "//xla/python/ifrt/ir:version",
        "//xla/python/ifrt/ir/export:vifrt",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TranslateLib",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:register",
        "@stablehlo//:version",
    ],
)
