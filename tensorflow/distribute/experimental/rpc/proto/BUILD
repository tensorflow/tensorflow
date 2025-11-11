load("@rules_python//python:proto.bzl", "py_proto_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/distribute/experimental/rpc:__subpackages__",
        "//tensorflow/python/distribute/experimental/rpc:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "tf_rpc_service_proto",
    srcs = ["tf_rpc_service.proto"],
    has_services = 1,
    create_go_proto = False,
    create_grpc_library = True,
    create_java_proto = False,
    create_service = True,
    protodeps = tf_additional_all_protos(),
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tf_rpc_service_py_pb2",
#     deps = [":tf_rpc_service_proto"],
# )
# copybara:uncomment_end
