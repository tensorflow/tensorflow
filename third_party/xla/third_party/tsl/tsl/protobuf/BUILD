load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)
load(
    "//tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core:__subpackages__",
        "@local_xla//xla/tsl:internal",
        "//tensorflow_models:__subpackages__",
    ]),
    features = if_google(["-parse_headers"]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["error_codes.proto"],
    make_default_target_header_only = True,
    protodeps = if_google(["//google/protobuf:any"]),
    visibility = ["//visibility:public"],
)
