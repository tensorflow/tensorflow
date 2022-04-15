load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)

package(
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
    cc_api_version = 2,
    cc_grpc_version = 1,
    cc_stubby_versions = ["2"],
    create_go_proto = False,
    create_java_proto = False,
    create_service = True,
    protodeps = tf_additional_all_protos(),
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tf_rpc_service_py_pb2",
#     api_version = 2,
#     deps = [":tf_rpc_service_proto"],
# )
# copybara:uncomment_end
