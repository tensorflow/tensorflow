load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        "-name=IfrtIrImport",
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
        "ifrt_copy_constants_pass.cc",
        "ifrt_dce_pass.cc",
        "ifrt_dump_atom_programs_pass.cc",
        "ifrt_duplicated_callee_elimination_pass.cc",
        "ifrt_insert_copy_arrays_for_returned_many_times_pass.cc",
        "ifrt_merge_ops_pass.cc",
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
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:utils",
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
