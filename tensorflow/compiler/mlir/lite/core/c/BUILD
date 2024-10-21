load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(["builtin_op_data.h"])

# LINT.IfChange(common)
cc_library(
    name = "tflite_common",
    srcs = [],
    hdrs = [
        "builtin_op_data.h",
        "dimension_type.h",
        "tflite_types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//tensorflow/lite/core/c:common)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "builtin_op_data_test",
    size = "small",
    srcs = ["builtin_op_data_test.cc"],
    extra_copts = ["-Wno-unused-variable"],
    deps = [
        ":tflite_common",
        "@com_google_googletest//:gtest_main",
    ],
)
