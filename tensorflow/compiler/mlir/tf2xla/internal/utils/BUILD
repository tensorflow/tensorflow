load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dialect_detection_utils",
    srcs = [
        "dialect_detection_utils.cc",
    ],
    hdrs = [
        "dialect_detection_utils.h",
    ],
    deps = [
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

tf_cc_test(
    name = "dialect_detection_utils_test",
    srcs = ["dialect_detection_utils_test.cc"],
    deps = [
        ":dialect_detection_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:chlo_ops",
    ],
)
