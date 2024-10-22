load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/python/ifrt:friends"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=IfrtIr",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "ifrt_duplicated_callee_elimination_pass.cc",
        "ifrt_lower_sharding_to_xla_pass.cc",
        "ifrt_merge_reshards_pass.cc",
        "ifrt_outline_atom_program_to_module_pass.cc",
        "ifrt_populate_atom_program_metadata_pass.cc",
        "ifrt_remove_ifrt_attrs_pass.cc",
        "ifrt_reshard_to_copy_arrays_pass.cc",
        "ifrt_verify_donation_pass.cc",
        "ifrt_verify_sharding_specified_pass.cc",
        "passes.cc",
        "spmd_expandable_interface_verification_pass.cc",
        "spmd_expansion_pass.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        ":utils",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/support:sharding_conversions",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "built_in_spmd_expansions",
    srcs = ["built_in_spmd_expansions.cc"],
    hdrs = ["built_in_spmd_expansions.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/python/ifrt/ir/transforms/spmd_expanders:spmd_expander",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/python/ifrt/ir",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
