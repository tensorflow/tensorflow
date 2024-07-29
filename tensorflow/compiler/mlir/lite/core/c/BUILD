load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(common)
cc_library(
    name = "common",
    srcs = [],
    hdrs = ["builtin_op_data.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tensorflow/lite/core/c:common)

# LINT.IfChange(private_common)
# This is a private target, its visibility is set to public only to be
# used by "tflite_custom_c_library" and "tflite_flex_cc_library".
# Do not use this target directly and don't consider it as a part of the public API.
alias(
    name = "private_common",
    actual = ":common",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)
# LINT.ThenChange(//tensorflow/lite/core/c:private_common)
