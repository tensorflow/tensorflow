load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

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
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:__pkg__",
    ],
)
# LINT.ThenChange(//tensorflow/lite/kernels/BUILD)
