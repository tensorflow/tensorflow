load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla/api/v0:__subpackages__",
    ],
)

cc_library(
    name = "mlir_pass_instrumentation",
    srcs = ["mlir_pass_instrumentation.cc"],
    hdrs = ["mlir_pass_instrumentation.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "mlir_pass_instrumentation_test",
    srcs = ["mlir_pass_instrumentation_test.cc"],
    deps = [
        ":mlir_pass_instrumentation",
        "//tensorflow/compiler/mlir/tf2xla/api/v0:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/core:test",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)
