load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

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
        "-name=IfrtIr",
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
        "ifrt_add_control_dependencies.cc",
        "ifrt_dump_atom_programs_pass.cc",
        "ifrt_duplicated_callee_elimination_pass.cc",
        "ifrt_insert_copy_arrays_for_returned_many_times_pass.cc",
        "ifrt_merge_copies_and_reshards_pass.cc",
        "ifrt_outline_atom_program_to_module_pass.cc",
        "ifrt_populate_atom_program_metadata_pass.cc",
        "ifrt_reshard_to_copy_arrays_pass.cc",
        "ifrt_verify_donation_pass.cc",
        "ifrt_verify_sharding_specified_pass.cc",
        "passes.cc",
        # go/keep-sorted end
    ],
    hdrs = [
        "passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        ":utils",
        "//xla/python/ifrt/ir",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//xla/python/ifrt/ir:friends",
    ]),
    deps = [
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/mlir/utils:type_util",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir/support:module_parsing",
        "//xla/python/ifrt/ir/support:sharding_conversions",
        "//xla/python/pjrt_ifrt:pjrt_dtype",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:device_assignment",
        "//xla/service/spmd/shardy:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "debug",
    srcs = ["debug.cc"],
    hdrs = ["debug.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:random",
        "@tsl//tsl/platform:regexp",
    ],
)

xla_cc_test(
    name = "debug_test",
    srcs = ["debug_test.cc"],
    deps = [
        ":debug",
        "//xla/python/ifrt/ir/support:module_parsing",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//xla/python/ifrt/ir/support:module_parsing",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)
