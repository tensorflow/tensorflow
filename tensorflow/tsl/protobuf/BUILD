load(
    "//tensorflow:tensorflow.bzl",
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
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    make_default_target_header_only = True,
    protodeps = [":dnn_proto"],
)

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["error_codes.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "protos_all",
    create_go_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
    ],
    visibility = ["//visibility:public"],
)
