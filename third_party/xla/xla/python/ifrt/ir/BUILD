load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

td_library(
    name = "ifrt_td",
    srcs = [
        "ifrt_dialect.td",
        "ifrt_interfaces.td",
        "ifrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "ifrt_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=ifrt",
            ],
            "ifrt_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=ifrt",
            ],
            "ifrt_dialect.cc.inc",
        ),
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=ifrt",
            ],
            "ifrt_types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=ifrt",
            ],
            "ifrt_types.cc.inc",
        ),
        (
            [
                "-gen-attrdef-decls",
                "--attrdefs-dialect=ifrt",
            ],
            "ifrt_attrs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "--attrdefs-dialect=ifrt",
            ],
            "ifrt_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_dialect.td",
    test = True,
    deps = [":ifrt_td"],
)

gentbl_cc_library(
    name = "ifrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ifrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ifrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_ops.td",
    test = True,
    deps = [":ifrt_td"],
)

gentbl_cc_library(
    name = "ifrt_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "ifrt_attr_interfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "ifrt_attr_interfaces.cc.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "ifrt_op_interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ifrt_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ifrt_interfaces.td",
    test = True,
    deps = [":ifrt_td"],
)

cc_library(
    name = "ir",
    srcs = [
        "ifrt_dialect.cc",
        "ifrt_interfaces.cc",
        "ifrt_ops.cc",
    ],
    hdrs = [
        "constants.h",
        "ifrt_dialect.h",
        "ifrt_interfaces.h",
        "ifrt_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":ifrt_dialect_inc_gen",
        ":ifrt_interfaces_inc_gen",
        ":ifrt_ops_inc_gen",
        ":sharding_param",
        "//xla/python/ifrt",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",  # buildcleaner: keep
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "sharding_param",
    srcs = ["sharding_param.cc"],
    hdrs = ["sharding_param.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "ifrt_ir_program",
    srcs = ["ifrt_ir_program.cc"],
    hdrs = ["ifrt_ir_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla/python/ifrt",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ifrt_ir_program_serdes",
    srcs = ["ifrt_ir_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":ifrt_ir_program",
        "//xla/mlir/utils:error_util",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt/support:module_parsing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "ifrt_ir_program_serdes_test",
    srcs = ["ifrt_ir_program_serdes_test.cc"],
    deps = [
        ":ifrt_ir_program",
        ":ifrt_ir_program_serdes",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt/support:module_parsing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "atom_program_compiler",
    hdrs = ["atom_program_compiler.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":ir",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)
