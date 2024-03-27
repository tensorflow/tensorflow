# copybara:uncomment_begin(oss-unused)
# load("//net/grpc:cc_grpc_library.bzl", "cc_grpc_library")
# load("//net/grpc/go/build_defs:go_grpc_library.bzl", "go_grpc_library")
# copybara:uncomment_end

# For platform specific build config
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__subpackages__",
        "//tensorflow_models:__subpackages__",
    ],
    features = if_google(["-parse_headers"]),
    licenses = ["notice"],
)

COMMON_PROTO_SRCS = [
    "config.proto",
    "cluster.proto",
    "debug.proto",
    "device_filters.proto",
    "device_properties.proto",
    "queue_runner.proto",
    "rewriter_config.proto",
    "tensor_bundle.proto",
    "saver.proto",
    "status.proto",
    "verifier_config.proto",
]

[
    [
        tf_pyclif_proto_library(
            name = "%s_pyclif" % proto_name,
            proto_lib = ":for_core_protos",
            proto_srcfile = "%s.proto" % proto_name,
            visibility = ["//visibility:public"],
        ),
    ]
    for proto_name in [
        "config",
        "device_properties",
        "meta_graph",
        "saved_model",
        "tensorflow_server",
    ]
]

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_xla//xla:autotuning_proto"],
    exports = ["@local_xla//xla:autotuning_proto"],
)

tf_proto_library(
    name = "conv_autotuning_proto",
    srcs = ["conv_autotuning.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/protobuf:dnn_proto",
    ],
)

tf_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = if_google(
        ["//tensorflow:internal"],
        ["//visibility:public"],
    ),
)

tf_proto_library(
    name = "worker_service_proto",
    srcs = ["worker_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    create_service = True,
    protodeps = [":worker_proto"],
)

tf_proto_library(
    name = "master_proto",
    srcs = ["master.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//tensorflow:internal"],
)

tf_proto_library(
    name = "master_service_proto",
    srcs = ["master_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    create_service = True,
    protodeps = [":master_proto"],
)

tf_proto_library(
    name = "eager_service_proto",
    srcs = ["eager_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    create_grpc_library = True,
    create_service = True,
    protodeps = tf_additional_all_protos(),
)

tf_proto_library(
    name = "replay_log_proto",
    srcs = ["replay_log.proto"],
    cc_api_version = 2,
    protodeps = [
        ":master_proto",
    ] + tf_additional_all_protos(),
)

tf_proto_library(
    name = "error_codes_proto_impl",
    srcs = ["error_codes.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_tsl//tsl/protobuf:error_codes_proto_impl"],
    exports = ["@local_tsl//tsl/protobuf:error_codes_proto_impl"],
)

exports_files(
    srcs = ["error_codes.proto"] + COMMON_PROTO_SRCS + [
        # Protos which are not needed on mobile builds, but should be included
        # in protos_all.
        #
        # Note that some protos are in neither core_proto_srcs nor this
        # filegroup; e.g. ones with individual proto_library targets.
        "control_flow.proto",
        # TODO(ebrevdo): Re-enable once CriticalSection is in core.
        # "critical_section.proto",
        "snapshot.proto",
        "data_service.proto",
        "service_config.proto",
        "debug_event.proto",
        "composite_tensor_variant.proto",
        "meta_graph.proto",
        "named_tensor.proto",
        "remote_tensor_handle.proto",
        "saved_model.proto",
        "saved_object_graph.proto",
        "struct.proto",
        "tensorflow_server.proto",
        "trackable_object_graph.proto",
        "transport_options.proto",
        "core_platform_payloads.proto",
        "fingerprint.proto",
    ],
)

tf_proto_library(
    name = "for_core_protos",
    srcs = COMMON_PROTO_SRCS + [
        # Protos which are not needed on mobile builds, but should be included
        # in protos_all.
        #
        # Note that some protos are in neither core_proto_srcs nor this
        # filegroup; e.g. ones with individual proto_library targets.
        "bfc_memory_map.proto",
        "control_flow.proto",
        # TODO(ebrevdo): Re-enable once CriticalSection is in core.
        # "critical_section.proto",
        # TODO: Move snapshot.proto and service_config.proto to a separate package.
        # NOTE: Creating an alias and adding the files does not work in OSS.
        # NOTE: tf_proto_library requires files to be in the same package.
        "snapshot.proto",
        "data_service.proto",
        "service_config.proto",
        "debug_event.proto",
        "composite_tensor_variant.proto",
        "meta_graph.proto",
        "named_tensor.proto",
        "remote_tensor_handle.proto",
        "rpc_options.proto",
        "saved_model.proto",
        "saved_object_graph.proto",
        "struct.proto",
        "tensorflow_server.proto",
        "trackable_object_graph.proto",
        "transport_options.proto",
        "core_platform_payloads.proto",
        "fingerprint.proto",
    ],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        "//tensorflow/core/framework:protos_all",
        "@local_tsl//tsl/protobuf:bfc_memory_map_proto",
        "@local_tsl//tsl/protobuf:coordination_config_proto",
        "@local_tsl//tsl/protobuf:rpc_options_proto",
        "@local_tsl//tsl/protobuf:status_proto",
    ],
    tags = ["alt_dep=//third_party/tensorflow/core:protos_all"],
    visibility = ["//visibility:public"],
    exports = [
        "@local_tsl//tsl/protobuf:bfc_memory_map_proto",
        "@local_tsl//tsl/protobuf:rpc_options_proto",
        "@local_tsl//tsl/protobuf:status_proto",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_grpc_library(
#     name = "worker_service_cc_grpc_proto",
#     srcs = [":worker_service_proto"],
#     generate_mocks = True,
#     service_namespace = "grpc_gen",
#     deps = [":worker_service_proto_cc"],
# )
#
# cc_grpc_library(
#     name = "master_service_cc_grpc_proto",
#     srcs = [":master_service_proto"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     generate_mocks = True,
#     service_namespace = "grpc_gen",
#     deps = [":master_service_proto_cc"],
# )
# copybara:uncomment_end
