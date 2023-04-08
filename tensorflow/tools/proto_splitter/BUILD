# Description:
#   Utilities for splitting and joining large protos > 2GB.

load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "chunk_proto",
    srcs = ["chunk.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [":versions_proto"],
)

# copybara:uncomment_begin(google-only)
#
# py_proto_library(
#     name = "versions_proto_py_pb2",
#     api_version = 2,
#     deps = [
#         ":versions_proto",
#     ],
# )
#
# py_proto_library(
#     name = "chunk_proto_py_pb2",
#     api_version = 2,
#     deps = [
#         ":chunk_proto",
#     ],
# )
#
# copybara:uncomment_end
