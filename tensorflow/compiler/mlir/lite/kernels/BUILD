load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(cc_library_padding)
cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/compiler/mlir/lite/kernels:__pkg__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ],
)
# LINT.ThenChange(//tensorflow/lite/kernels/BUILD)

# LINT.IfChange(cc_library_kernel_util)
cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = [
        "kernel_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/compiler/mlir/lite:kernel_api",
        "//tensorflow/compiler/mlir/lite/core/c:common",
    ],
)
# LINT.ThenChange(//tensorflow/lite/kernels/BUILD)
