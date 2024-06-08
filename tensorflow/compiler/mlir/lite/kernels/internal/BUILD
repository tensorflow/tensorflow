load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility_macros",
    hdrs = ["compatibility_macros.h"],
    compatible_with = get_compatible_with_portable(),
)

# LINT.IfChange
cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":compatibility_macros",
        "//tensorflow/compiler/mlir/lite/core:macros",
        "@gemmlowp//:fixedpoint",
    ],
)

# LINT.ThenChange(//tensorflow/lite/kernels/internal/BUILD)
