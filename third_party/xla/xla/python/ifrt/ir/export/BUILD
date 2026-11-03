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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/python/ifrt/ir:internal"]),
    licenses = ["notice"],
)

td_library(
    name = "vifrt_td",
    srcs = [
        "vifrt_dialect.td",
        "vifrt_interfaces.td",
        "vifrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "vifrt_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_attr_interfaces.h.inc": ["-gen-attr-interface-decls"],
        "vifrt_attr_interfaces.cc.inc": ["-gen-attr-interface-defs"],
        "vifrt_type_interfaces.h.inc": ["-gen-type-interface-decls"],
        "vifrt_type_interfaces.cc.inc": ["-gen-type-interface-defs"],
        "vifrt_op_interfaces.h.inc": ["-gen-op-interface-decls"],
        "vifrt_op_interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_interfaces.td",
    test = True,
    deps = [":vifrt_td"],
)

gentbl_cc_library(
    name = "vifrt_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=vifrt",
        ],
        "vifrt_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=vifrt",
        ],
        "vifrt_types.h.inc": [
            "-gen-typedef-decls",
            "--typedefs-dialect=vifrt",
        ],
        "vifrt_types.cc.inc": [
            "-gen-typedef-defs",
            "--typedefs-dialect=vifrt",
        ],
        "vifrt_attrs.h.inc": [
            "-gen-attrdef-decls",
            "--attrdefs-dialect=vifrt",
        ],
        "vifrt_attrs.cc.inc": [
            "-gen-attrdef-defs",
            "--attrdefs-dialect=vifrt",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_dialect.td",
    test = True,
    deps = [":vifrt_td"],
)

gentbl_cc_library(
    name = "vifrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "vifrt_ops.h.inc": ["-gen-op-decls"],
        "vifrt_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vifrt_ops.td",
    test = True,
    deps = [":vifrt_td"],
)

cc_library(
    name = "vifrt",
    srcs = [
        "vifrt_bytecode.cc",
        "vifrt_dialect.cc",
    ],
    hdrs = [
        "vifrt_bytecode.h",
        "vifrt_dialect.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":vifrt_dialect_inc_gen",
        ":vifrt_interfaces_inc_gen",
        ":vifrt_ops_inc_gen",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/ifrt/ir:version",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=IfrtIrExport",
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
        "ifrt_atom_programs_from_vhlo_pass.cc",
        "ifrt_atom_programs_to_vhlo_pass.cc",
        "ifrt_legalize_to_vifrt_pass.cc",
        "ifrt_remove_attrs_from_other_dialects_pass.cc",
        "passes.cc",
        "vifrt_legalize_to_ifrt_pass.cc",
        "vifrt_to_version_pass.cc",
        # go/keep-sorted end
    ],
    hdrs = [
        "map_ifrt_to_vifrt.h",
        "passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        ":vifrt",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program_proto_cc",
        "//xla/python/ifrt/ir:ifrt_ops_inc_gen",
        "//xla/python/ifrt/ir:version",
        "//xla/python/ifrt/ir/transforms:utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@tsl//tsl/platform:protobuf",
    ],
)
