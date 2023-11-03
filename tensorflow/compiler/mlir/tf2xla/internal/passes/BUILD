load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/pathways/serving/transforms:__pkg__",
        "//tensorflow/compiler/mlir:__pkg__",
        "//tensorflow/compiler/mlir/tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "clustering_passes",
    srcs = [
        "verify_clustering_pass.cc",
    ],
    hdrs = [
        "clustering_passes.h",
    ],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        ":extract_head_tail_outside_compilation",
        ":tpu_cluster_formation",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:string_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tpu_rewrite_device_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/transforms/toposort:Pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "clustering_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TFXLABridge",
            ],
            "clustering_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "clustering_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

tf_cc_test(
    name = "verify_clustering_pass_test",
    srcs = ["verify_clustering_pass_test.cc"],
    deps = [
        ":clustering_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:test_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

cc_library(
    name = "tpu_cluster_formation",
    srcs = ["tpu_cluster_formation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:string_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tpu_rewrite_device_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_pass_inc_gen",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "extract_head_tail_outside_compilation",
    srcs = ["extract_head_tail_outside_compilation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:string_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:tpu_rewrite_device_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "dialect_to_executor_passes",
    srcs = [
        "dialect_to_executor_passes.h",
    ],
    textual_hdrs = [
        "dialect_to_executor_passes.h.inc",
    ],
    deps = [
        ":dialect_to_executor_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core/transforms/toposort:Pass",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "dialect_to_executor_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TFXLABridge",
            ],
            "dialect_to_executor_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect_to_executor_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
