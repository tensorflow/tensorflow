load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "tflite_types",
    srcs = [
        "tflite_types.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

exports_files(
    srcs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

# LINT.IfChange(common)
cc_library(
    name = "tflite_common",
    srcs = [],
    hdrs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
        "//tensorflow/lite/core/c:__subpackages__",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tensorflow/lite/core/c:common)
