load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

proto_library(
    name = "benchmark_result_proto",
    srcs = ["benchmark_result.proto"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "benchmark_result",  # bzl adds _py for python proto library
    srcs = ["benchmark_result.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "benchmark_result_py_pb2",
#     compatible_with = get_compatible_with_portable(),
#     deps = [":benchmark_result_proto"],
# )
# copybara:uncomment_end
