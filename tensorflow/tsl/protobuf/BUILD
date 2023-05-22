load(
    "//tensorflow/tsl:tsl.bzl",
    "if_google",
    "set_external_visibility",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tsl:internal",
        "//tensorflow_models:__subpackages__",
    ]),
    features = if_google(["-parse_headers"]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "bfc_memory_map_proto",
    srcs = ["bfc_memory_map.proto"],
    make_default_target_header_only = True,
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
    name = "status_proto",
    srcs = ["status.proto"],
    make_default_target_header_only = True,
    protodeps = [":error_codes_proto_impl"],
)

tf_proto_library(
    name = "histogram_proto",
    srcs = ["histogram.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "coordination_config_proto",
    srcs = ["coordination_config.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "coordination_service_proto",
    srcs = ["coordination_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    create_grpc_library = True,
    create_java_proto = False,
    create_service = True,
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "coordination_service_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":coordination_service_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "distributed_runtime_payloads_proto",
    srcs = ["distributed_runtime_payloads.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "rpc_options_proto",
    srcs = ["rpc_options.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "test_log_proto",
    srcs = ["test_log.proto"],
    make_default_target_header_only = True,
    visibility = set_external_visibility([
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tsl/util:__pkg__",
    ]),
)

tf_proto_library(
    name = "protos_all",
    create_go_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        # TODO(tlongeri): Conceptually, these fit into protos_all but adding them currently causes
        # breakages (and they are not actually used).
        # ":autotuning_proto",
        # ":dnn_proto",
        ":bfc_memory_map_proto",
        ":coordination_config_proto",
        ":distributed_runtime_payloads_proto",
        ":error_codes_proto_impl",
        ":histogram_proto",
        ":rpc_options_proto",
        ":status_proto",
        ":test_log_proto",
    ],
    visibility = ["//visibility:public"],
)
