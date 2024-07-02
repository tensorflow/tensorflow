load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "light_model_builder",
    srcs = ["light_model_builder.cc"],
    hdrs = ["light_model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:string",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)
