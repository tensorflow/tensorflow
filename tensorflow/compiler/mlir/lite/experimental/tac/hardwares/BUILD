load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_hardware",
    srcs = ["gpu_hardware.cc"],
    hdrs = ["gpu_hardware.h"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cpu_hardware",
    srcs = ["cpu_hardware.cc"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nnapi_hardware",
    srcs = ["nnapi_hardware.cc"],
    hdrs = ["nnapi_hardware.h"],
    deps = [
        ":target_hardware",
        "//tensorflow/compiler/mlir/lite:cost_estimators",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "//tensorflow/compiler/mlir/lite/experimental/tac/hardwares:simple_hardware",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "target_hardware",
    srcs = ["target_hardware.cc"],
    hdrs = ["target_hardware.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/experimental/tac:common",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "simple_hardware",
    srcs = ["simple_hardware.cc"],
    hdrs = ["simple_hardware.h"],
    deps = [
        ":target_hardware",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "all-target-hardwares",
    deps = [
        ":cpu_hardware",
        ":gpu_hardware",
        ":nnapi_hardware",
    ],
    alwayslink = 1,
)
