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
    tags = ["keep_dep"],
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
        "@llvm-project//llvm:AArch64CodeGen",  # build_cleaner: keep
        "@llvm-project//llvm:BitWriter",
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
)

xla_cc_test(
    name = "vector_ops_test",
    srcs = ["vector_ops_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_ir_header(
    name = "tanh_ll",
    src = "tanh.cc",
    deps = [
        ":cpp_context_provider",
        ":tanh",
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
    name = "atan",
    srcs = ["atan.cc"],
    hdrs = ["atan.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [":vector_ops"],
)

cc_ir_header(
    name = "atan_ll",
    src = "atan.cc",
    deps = [
        ":atan",
        ":cpp_context_provider",
        ":vector_ops",
    ],
)

cc_library(
    name = "ynnpack_unary_lib",
    srcs = ["ynnpack_unary.cc"],
    hdrs = ["ynnpack_unary.h"],
    copts = ["-Wno-psabi"],
    deps = [
        ":vector_ops",
        "@XNNPACK//ynnpack/base/simd",
    ],
)

cc_ir_header(
    name = "ynnpack_unary_ll",
    src = "ynnpack_unary.cc",
    copts = ["-fbuiltin"],
    # YNNPACK's gnu_vector backend implements fma as a per-lane std::fma loop,
    # which clang only vectorizes when the target has FMA. Without -mfma on
    # x86_64 the bitcode keeps scalar llvm.fma calls, and nothing re-vectorizes
    # them after inlining. Target features are stripped from the bitcode, so this
    # does not require FMA at runtime. Not needed once YNNPACK uses
    # __builtin_elementwise_fma.
    x86_64_copts = ["-mfma"],
    deps = [
        ":cpp_context_provider",
        ":vector_ops",
        ":ynnpack_unary_lib",
        "@XNNPACK//ynnpack/base/simd",
    ],
)

xla_cc_test(
    name = "cpp_gen_intrinsics_test",
    srcs = ["cpp_gen_intrinsics_test.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        ":atan_ll",
        "//xla/codegen/intrinsic:intrinsic_compiler_lib",
        "//xla/codegen/intrinsic:simple_jit_runner",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:denormal",
    ],
)

xla_cc_test(
    name = "atan_test",
    srcs = ["atan_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":atan",
        ":atan_ll",
        ":vector_ops",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "expm1",
    srcs = ["expm1.cc"],
    hdrs = ["expm1.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
    ],
)

cc_ir_header(
    name = "expm1_ll",
    src = "expm1.cc",
    deps = [
        ":cpp_context_provider",
        ":expm1",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "expm1_test",
    srcs = ["expm1_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":expm1",
        ":expm1_ll",
        ":vector_ops",
        "//xla:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "ynnpack_unary_test",
    srcs = ["ynnpack_unary_test.cc"],
    deps = [
        ":_cpp_gen_intrinsics",
        ":ynnpack_unary_ll",
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
        ":atan_ll",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "custom_cpp_intrinsics_lib",
    srcs = ["custom_cpp_intrinsics.cc"],
    hdrs = ["custom_cpp_intrinsics.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
    ],
)

cc_ir_header(
    name = "custom_cpp_intrinsics_ll",
    src = "custom_cpp_intrinsics.cc",
    deps = [
        ":cpp_context_provider",
        ":custom_cpp_intrinsics_lib",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "custom_cpp_intrinsics_test",
    srcs = ["custom_cpp_intrinsics_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":custom_cpp_intrinsics_lib",
        ":custom_cpp_intrinsics_ll",
        ":vector_ops",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "trig_lib",
    srcs = ["trig.cc"],
    hdrs = ["trig.h"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":vector_ops",
    ],
)

cc_ir_header(
    name = "trig_ll",
    src = "trig.cc",
    deps = [
        ":cpp_context_provider",
        ":trig_lib",
        ":vector_ops",
    ],
)

xla_cc_test(
    name = "trig_test",
    srcs = ["trig_test.cc"],
    copts = [
        "-Wno-psabi",
    ],
    deps = [
        ":_cpp_gen_intrinsics",
        ":trig_lib",
        ":trig_ll",
        ":vector_ops",
        "//xla/codegen/intrinsic:test_matchers",
        "@com_google_absl//absl/base",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "_cpp_gen_intrinsics",
    srcs = ["cpp_gen_intrinsics.cc"],
    hdrs = ["cpp_gen_intrinsics.h"],
    deps = [
        ":atan_ll",
        ":custom_cpp_intrinsics_ll",
        ":expm1_ll",
        ":trig_ll",
        ":ynnpack_unary_ll",
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
