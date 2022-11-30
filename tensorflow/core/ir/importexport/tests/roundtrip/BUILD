# Regression tests for bridge.
load(":roundtrip.bzl", "glob_roundtrip_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/ir/importexport:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "roundtrip",
    srcs = [
        "roundtrip.cc",
    ],
    hdrs = ["roundtrip.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//llvm:Support",
    ],
)

# This will run the `verify-roundtrip` binary below on every single pbtxt in this directory.
glob_roundtrip_tests(
    exclude = [],
    test_file_exts = ["pbtxt"],
)

tf_cc_binary(
    name = "verify-roundtrip",
    testonly = 1,
    srcs = ["verify_roundtrip.cc"],
    deps = [
        ":roundtrip",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",  # Ops need to be registered for import.
        "//tensorflow/core/ir/importexport:graphdef_export",
        "//tensorflow/core/ir/importexport:graphdef_import",
        "//tensorflow/core/ir/importexport:load_proto",
        "//tensorflow/core/transforms/consolidate_attrs:Pass",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
