# Description:
#   Contains Keras protobufs

# Placeholder: load py_proto_library
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "projector_config_proto",
    srcs = ["projector_config.proto"],
)

tf_proto_library(
    name = "saved_metadata_proto",
    srcs = ["saved_metadata.proto"],
    protodeps = [":versions_proto"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "saved_metadata_proto_py_pb2",
#     deps = [":saved_metadata_proto"],
# )
#
# py_proto_library(
#     name = "projector_config_proto_py_pb2",
#     deps = [":projector_config_proto"],
# )
#
# py_proto_library(
#     name = "versions_proto_py_pb2",
#     deps = [":versions_proto"],
# )
# copybara:uncomment_end
