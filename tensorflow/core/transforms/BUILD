load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        ":__subpackages__",
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "--name",
                "TFGraph",
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
    name = "TopoSortPass",
    srcs = ["toposort/toposort_pass.cc"],
    hdrs = ["toposort/toposort_pass.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "GraphToFunc",
    srcs = ["graph_to_func/graph_to_func.cc"],
    hdrs = ["graph_to_func/graph_to_func.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "GraphToFuncPass",
    srcs = ["graph_to_func/graph_to_func_pass.cc"],
    hdrs = ["graph_to_func/graph_to_func_pass.h"],
    deps = [
        ":GraphToFunc",
        ":PassIncGen",
        "//tensorflow/core/ir:Dialect",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "PassRegistration",
    hdrs = ["pass_registration.h"],
    deps = [
        ":GraphToFuncPass",
        ":PassIncGen",
        ":TopoSortPass",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
tf_native_cc_binary(
    name = "tfg-transforms-opt",
    srcs = ["tfg-transforms-opt.cc"],
    deps = [
        ":PassRegistration",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TransformUtils",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-transforms-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
