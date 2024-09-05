load("//tensorflow:tensorflow.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "mla_test_utils",
    testonly = True,
    hdrs = ["mla_test_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive/google:google_env",
        "//tensorflow/cc/experimental/tfa:saved_model_converter",
        "//tensorflow/cc/experimental/tfa:test_utils",
    ]),
)

cc_library(
    name = "mla_utils",
    hdrs = ["mla_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive:status_macro",
        "//tensorflow/cc/experimental/tfa:saved_model_converter",
        "//third_party/mira/mlarchive/google:google_env",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
    ]),
)
