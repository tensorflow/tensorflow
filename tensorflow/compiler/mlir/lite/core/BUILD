load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts_warnings")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "model_builder_base.h",
    ],
    visibility = ["//tensorflow/lite/core:__pkg__"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_builder_base",
    srcs = ["model_builder_base.cc"],
    hdrs = ["model_builder_base.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/core:__pkg__",
    ],
    deps = [
        ":macros",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "//tensorflow/compiler/mlir/lite/core/api:verifier",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "absl_error_model_builder",
    srcs = ["absl_error_model_builder.cc"],
    hdrs = ["absl_error_model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
    ],
    deps = [
        ":model_builder_base",
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "@com_google_absl//absl/log",
    ],
)
