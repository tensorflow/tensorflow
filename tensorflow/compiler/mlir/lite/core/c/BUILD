load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(cc_library_common)
cc_library(
    name = "common",
    srcs = [],
    hdrs = ["builtin_op_data.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/lite/kernels:__pkg__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tensorflow/lite/core/c/BUILD)
