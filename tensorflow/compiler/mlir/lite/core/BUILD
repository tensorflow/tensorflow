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
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/api:verifier",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
    alwayslink = 1,
)
