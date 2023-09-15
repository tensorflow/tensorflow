# On-disk serialization format for TensorFlow models.

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows_or_mac",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":__subpackages__",
        "//smartass/brain/configure:__subpackages__",
        "//tensorflow/cc/experimental/tf2:__subpackages__",
        "//tensorflow/cc/experimental/tfa:__subpackages__",
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "internal_api",
    srcs = ["internal_api.cc"],
    hdrs = ["internal_api.h"],
    deps = [
        "//tensorflow/cc/saved_model:metrics",
        "//tensorflow/cc/saved_model:util",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/tools/proto_splitter/cc:max_size",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + if_not_windows_or_mac([
        "//tensorflow/tools/proto_splitter:merge",
        "//tensorflow/tools/proto_splitter/cc:saved_model_splitter",
    ]),
)
