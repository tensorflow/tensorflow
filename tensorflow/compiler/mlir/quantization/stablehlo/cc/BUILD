load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
