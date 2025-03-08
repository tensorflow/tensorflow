load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//platforms/darwinn/compiler:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "quantization_driver",
    srcs = [
        "quantization_driver.cc",
    ],
    hdrs = [
        "quantization_driver.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy:portable_tensor_utils",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)
