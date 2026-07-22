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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "stablehlo_to_hlo_to_stablehlo",
    srcs = ["stablehlo_to_hlo_to_stablehlo.cc"],
    hdrs = ["stablehlo_to_hlo_to_stablehlo.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate:register",
        "//xla/hlo/translate:stablehlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:UBDialect",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "testing_pipeline",
    srcs = ["testing_pipeline.cc"],
    hdrs = ["testing_pipeline.h"],
    deps = [
        ":stablehlo_to_hlo_to_stablehlo",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@shardy//shardy/dialect/sdy/transforms/import:passes",
    ],
)
