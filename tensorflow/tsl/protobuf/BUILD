load(
    "//tensorflow/tsl:tsl.bzl",
    "if_google",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__subpackages__",
        "//tensorflow_models:__subpackages__",
    ],
    features = if_google(["-parse_headers"]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["error_codes.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "histogram_proto",
    srcs = ["histogram.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "protos_all",
    create_go_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        ":histogram_proto",
    ],
    visibility = ["//visibility:public"],
)
