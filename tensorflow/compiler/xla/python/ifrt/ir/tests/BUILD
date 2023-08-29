load("//tensorflow/compiler/xla:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/xla:run_lit.sh",
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
)

cc_binary(
    name = "ifrt-opt",
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "//tensorflow/compiler/xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//tensorflow/compiler/xla/python/ifrt/ir/transforms:passes",
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
    visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "//tensorflow/compiler/xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:ref_count",
    ],
)

cc_library(
    name = "executable_impl_test_lib",
    testonly = True,
    srcs = ["executable_impl_test_lib.cc"],
    visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    deps = [
        ":executable_impl_test_base",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "//tensorflow/compiler/xla/python/ifrt/ir:compiler",
        "//tensorflow/compiler/xla/python/pjrt_ifrt:xla_ifrt",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:ref_count",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "executable_test_no_impl",
    srcs = [],
    deps = [
        ":executable_impl_test_lib",
        "//tensorflow/compiler/xla/python/ifrt:no_impl_test_main",
    ],
)
