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
    name = "profiling_info_proto",
    srcs = ["profiling_info.proto"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

proto_library(
    name = "model_runtime_info_proto",
    srcs = ["model_runtime_info.proto"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [":profiling_info_proto"],
)

tf_proto_library(
    name = "profiling_info",  # bzl adds _py
    srcs = ["profiling_info.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "model_runtime_info",  # bzl adds _py
    srcs = ["model_runtime_info.proto"],
    protodeps = [":profiling_info"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "profiling_info_py_pb2",
#     compatible_with = get_compatible_with_portable(),
#     deps = [":profiling_info_proto"],
# )
#
# py_proto_library(
#     name = "model_runtime_info_py_pb2",
#     compatible_with = get_compatible_with_portable(),
#     deps = [":model_runtime_info_proto"],
# )
# copybara:uncomment_end
