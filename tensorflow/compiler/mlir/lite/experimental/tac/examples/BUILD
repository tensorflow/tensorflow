load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "example_hardware",
    srcs = ["example_hardware.cc"],
    hdrs = ["example_hardware.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:simple_hardware",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "example-hardware-translate",
    deps = [
        ":example_hardware",
        "//tensorflow/compiler/mlir/lite/experimental/tac:tac-translate-lib",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:all-target-hardwares",
    ],
)
