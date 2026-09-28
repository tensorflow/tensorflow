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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_oss")
load("//xla/tsl/platform:rules_cc.bzl", "cc_binary", "cc_library")
load(
    "//xla/tsl/platform/default:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_hexagon_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load(":cc_to_llvm_ir.bzl", "cc_ir_header")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

config_setting(
    name = "x86_64",
    constraint_values = ["@platforms//cpu:x86_64"],
)

# This empty target provides the C++ compilation context for the C++ source file.
cc_library(
    name = "cpp_context_provider",
    srcs = [],
)

cc_binary(
    name = "embed_bitcode",
    srcs = ["embed_bitcode.cc"],
    defines = if_llvm_aarch32_available(["TF_LLVM_AARCH32_AVAILABLE=1"]) + [
        "TF_LLVM_AARCH64_AVAILABLE=1",
    ] + if_llvm_hexagon_available([
        "TF_LLVM_HEXAGON_AVAILABLE=1",
    ]) + if_llvm_powerpc_available([
        "TF_LLVM_POWERPC_AVAILABLE=1",
    ]) + if_llvm_system_z_available([
        "TF_LLVM_S390X_AVAILABLE=1",
    ]) + if_llvm_x86_available([
        "TF_LLVM_X86_AVAILABLE=1",
    ]),
    deps = [
        "//xla/util:embedded_constant_buffers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ] + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMCodeGen",
    ]) + if_llvm_hexagon_available([
        "@llvm-project//llvm:HexagonCodeGen",
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",
    ]),
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
    ],
)

cc_library(
    name = "vector_ops",
    hdrs = ["vector_ops.h"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "tanh_ll",
    src = "tanh.cc",
    deps = [
        ":cpp_context_provider",
        ":tanh",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        ":tanh_ll",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "eigen_unary_lib",
    srcs = ["eigen_unary.cc"],
    hdrs = ["eigen_unary.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "eigen_unary_32_ll",
    src = "eigen_unary.cc",
    copts = if_oss(
        [
            "-DEIGEN_GENERIC_VECTOR_SIZE_BYTES=64",
            "-mprefer-vector-width=512",
        ],
        [
            "-DEIGEN_GENERIC_VECTOR_SIZE_BYTES=32",
            "-mprefer-vector-width=256",
        ],
    ),
    deps = [
        ":cpp_context_provider",
        ":eigen_unary_lib",
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

cc_ir_header(
    name = "eigen_unary_64_ll",
    src = "eigen_unary.cc",
    copts = [
        "-DEIGEN_GENERIC_VECTOR_SIZE_BYTES=64",
        "-mprefer-vector-width=512",
    ] + if_oss(
        [],
        select({
            ":x86_64": ["-mavx512f"],
            "//conditions:default": [],
        }),
    ),
    deps = [
        ":cpp_context_provider",
        ":eigen_unary_lib",
        ":vector_ops",
        "@eigen_archive//:eigen3",
    ],
)

xla_cc_test(
    name = "eigen_unary_test",
    srcs = ["eigen_unary_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":eigen_unary_32_ll",
        ":eigen_unary_64_ll",
        ":eigen_unary_lib",
        ":vector_ops",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "dump_llvm_ir",
    srcs = ["dump_llvm_ir.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        "//xla/codegen/intrinsic",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "_cpp_gen_intrinsics",
    srcs = ["cpp_gen_intrinsics.cc"],
    hdrs = ["cpp_gen_intrinsics.h"],
    deps = [
        ":eigen_unary_32_ll",
        ":eigen_unary_64_ll",
        "//xla/codegen/intrinsic",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "_intrinsic_declarations",
    hdrs = ["intrinsic_declarations.h"],
    deps = [
        ":_cpp_gen_intrinsics",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:type",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:ir_headers",
    ],
)
