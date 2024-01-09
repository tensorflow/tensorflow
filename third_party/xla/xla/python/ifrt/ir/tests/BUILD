load("//xla:glob_lit_test.bzl", "glob_lit_tests")
load("//xla:xla.bzl", "xla_cc_test")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "ifrt-opt",
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//xla/python/ifrt/ir/transforms:passes",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status_macros",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "executable_impl_test_lib",
    testonly = True,
    srcs = ["executable_impl_test_lib.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":executable_impl_test_base",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:compiler",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/lib/core:status_test_util",
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
