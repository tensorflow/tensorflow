# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_pyclif_proto_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__subpackages__",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

COMMON_PROTO_SRCS = [
    "bfc_memory_map.proto",
    "config.proto",
    "cluster.proto",
    "debug.proto",
    "device_filters.proto",
    "device_properties.proto",
    "graph_debug_info.proto",
    "queue_runner.proto",
    "rewriter_config.proto",
    "tensor_bundle.proto",
    "saver.proto",
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
        "graph_debug_info",
        "meta_graph",
        "saved_model",
    ]
]

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "conv_autotuning_proto",
    srcs = ["conv_autotuning.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/stream_executor:dnn_proto",
    ],
)

tf_proto_library_cc(
    name = "worker_proto",
    srcs = ["worker.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)

tf_proto_library_cc(
    name = "worker_service_proto",
    srcs = ["worker_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    protodeps = [":worker_proto"],
)

tf_proto_library_cc(
    name = "master_proto",
    srcs = ["master.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//tensorflow:internal"],
)

tf_proto_library_cc(
    name = "master_service_proto",
    srcs = ["master_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_stubby_versions = ["2"],
    protodeps = [":master_proto"],
)

tf_proto_library_cc(
    name = "eager_service_proto",
    srcs = ["eager_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    cc_stubby_versions = ["2"],
    protodeps = tf_additional_all_protos(),
)

tf_proto_library_cc(
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
    cc_api_version = 2,
    make_default_target_header_only = True,
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
        "data/experimental/snapshot.proto",
        "debug_event.proto",
        "meta_graph.proto",
        "named_tensor.proto",
        "remote_tensor_handle.proto",
        "saved_model.proto",
        "saved_object_graph.proto",
        "struct.proto",
        "tensorflow_server.proto",
        "trackable_object_graph.proto",
        "transport_options.proto",
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
        "control_flow.proto",
        # TODO(ebrevdo): Re-enable once CriticalSection is in core.
        # "critical_section.proto",
        "data/experimental/snapshot.proto",
        "debug_event.proto",
        "meta_graph.proto",
        "named_tensor.proto",
        "remote_tensor_handle.proto",
        "saved_model.proto",
        "saved_object_graph.proto",
        "struct.proto",
        "tensorflow_server.proto",
        "trackable_object_graph.proto",
        "transport_options.proto",
    ],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":error_codes_proto_impl",
        "//tensorflow/core/framework:protos_all",
    ],
)
