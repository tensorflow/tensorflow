load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core:__subpackages__",
        "//xla/tsl:internal",
        "//tensorflow_models:__subpackages__",
    ]),
    features = if_google(["-parse_headers"]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "bfc_memory_map_proto",
    srcs = ["bfc_memory_map.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "test_log_proto",
    srcs = ["test_log.proto"],
    make_default_target_header_only = True,
    protodeps = if_google([
        "@com_google_protobuf//:any",
        "@com_google_protobuf//:wrappers",
    ]),
    visibility = ["//visibility:public"],
)
