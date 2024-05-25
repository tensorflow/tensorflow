# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

proto_library(
    name = "profiling_info_proto",
    srcs = ["profiling_info.proto"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "profiling_info_cc_proto",
    compatible_with = get_compatible_with_portable(),
    deps = [":profiling_info_proto"],
)

tf_proto_library(
    name = "profiling_info",  # bzl adds _py
    srcs = ["profiling_info.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "profiling_info_py_pb2",
#     api_version = 2,
#     compatible_with = get_compatible_with_portable(),
#     deps = [":profiling_info_proto"],
# )
# copybara:uncomment_end
