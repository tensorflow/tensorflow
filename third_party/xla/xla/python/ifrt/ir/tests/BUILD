load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "ifrt_compile_atom_program.mlir",
            "ifrt_compile_and_propagate_shardings.mlir",
            "ifrt_duplicated_callee_elimination.mlir",
            "ifrt_lower_mpmd_reshard_to_call.mlir",
            "ifrt_lower_sharding_to_xla.mlir",
            "ifrt_merge_reshards.mlir",
            "ifrt_outline_atom_program_to_module.mlir",
            "ifrt_populate_atom_program_metadata.mlir",
            "ifrt_remove_ifrt_attrs.mlir",
            "ifrt_reshard_to_copy_arrays.mlir",
            "ifrt_verify_donation.mlir",
            "ifrt_verify_sharding_specified.mlir",
            "spmd_expansion.mlir",
            "spmd_interface_verification.mlir",
            "verify_array.mlir",
            "verify_assemble.mlir",
            "verify_attrs.mlir",
            "verify_call.mlir",
            "verify_call_loaded_executable.mlir",
            "verify_copy_arrays.mlir",
            "verify_disassemble.mlir",
            "verify_loaded_executable.mlir",
            "verify_remap_arrays.mlir",
            "verify_reshard.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "ifrt-opt",
    testonly = True,
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:atom_program_compiler",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/support:module_parsing",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

cc_library(
    name = "executable_impl_test_base",
    testonly = True,
    srcs = ["executable_impl_test_base.cc"],
    hdrs = ["executable_impl_test_base.h"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "executable_impl_test_lib",
    testonly = True,
    srcs = ["executable_impl_test_lib.cc"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        ":executable_impl_test_base",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "executable_test_no_impl",
    srcs = [],
    deps = [
        ":executable_impl_test_lib",
        "//xla/python/ifrt:no_impl_test_main",
    ],
)
