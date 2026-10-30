# Copyright 2026 The OpenXLA Authors.
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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/python/ifrt/ir:internal"]),
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=IfrtIrCompilation",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        # go/keep-sorted start
        "ifrt_compile_atom_program_pass.cc",
        "ifrt_lower_atom_program_metadata_to_xla_pass.cc",
        "ifrt_lower_mpmd_reshard_to_call_pass.cc",
        "ifrt_precompile_atom_program_preprocessing_pass.cc",
        "ifrt_remove_ifrt_attrs_pass.cc",
        "ifrt_to_dot_pass.cc",
        "ifrt_verify_bound_external_loaded_executable_pass.cc",
        "ifrt_verify_device_type_consistency_pass.cc",
        "passes.cc",
        # go/keep-sorted end
    ],
    hdrs = [
        "passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":atom_program_compiler",
        ":passes_inc_gen",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:user_context",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir/support:sharding_conversions",
        "//xla/python/ifrt/ir/transforms:debug",
        "//xla/python/ifrt/ir/transforms:utils",
        "//xla/python/ifrt/mlir:fingerprint_utils",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:random",
    ],
)

cc_library(
    name = "atom_program_compiler",
    hdrs = ["atom_program_compiler.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//xla/python/ifrt/ir:friends"]),
    deps = [
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "basic_atom_program_compiler",
    srcs = ["basic_atom_program_compiler.cc"],
    hdrs = ["basic_atom_program_compiler.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility(["//xla/python/ifrt/ir:friends"]),
    deps = [
        ":atom_program_compiler",
        ":utils",
        "//xla:status_macros",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:device_assignment",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

exports_files(
    ["utils.h"],
    visibility = if_google(
        ["//xla/python/ifrt/ir/google:__pkg__"],
        ["//visibility:private"],
    ),
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        if_google("//xla/python/ifrt/ir/google:compilation_utils", ":utils_oss"),
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "utils_oss",
    srcs = [
        "utils.cc",
        "utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = True,
)

cc_library(
    name = "compiled_ifrt_ir_program",
    srcs = ["compiled_ifrt_ir_program.cc"],
    hdrs = ["compiled_ifrt_ir_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//xla/python/ifrt/ir:friends"]),
    deps = [
        ":atom_program_compiler",
        ":passes",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:program_interpreter",
        "//xla/python/ifrt/ir:utils",
        "//xla/python/ifrt/ir/transforms:debug",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/ir/transforms:utils",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility(["//xla/python/ifrt/ir:friends"]),
    deps = [
        ":atom_program_compiler",
        ":compiled_ifrt_ir_program",
        "//xla:util",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir:ifrt_ir_executable_version",
        "//xla/python/ifrt/ir:ifrt_ir_loaded_executable",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:serialization_utils",
        "//xla/python/ifrt/ir:version",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)
