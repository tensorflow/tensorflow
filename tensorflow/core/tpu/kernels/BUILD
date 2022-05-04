# TPU Kernel Implementations

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//tensorflow:tensorflow.bzl", "if_libtpu", "tf_cc_test", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")  # buildifier: disable=same-origin-load
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

# Config setting to enable go/libtpu support.

package(
    default_visibility = [
        "//tensorflow/compiler/xrt/kernels:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
        "//tensorflow/stream_executor/tpu:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":cross_replica_ops",
        ":host_compute_ops",
        ":image_resize_ops",
        ":infeed_ops",
        ":outfeed_ops",
        ":replication_ops",
        ":sharding_util_ops",
        ":topk_ops",
        ":tpu_compile_op",
        ":tpu_configuration_ops",
        ":tpu_embedding_configuration_ops",
        ":tpu_embedding_enqueue_ops",
        ":tpu_embedding_load_retrieve_ops",
        ":tpu_embedding_ops",
        ":tpu_execute_op",
        ":tpu_functional_ops",
        ":tpu_handle_to_key_op",
        ":tpu_ordinal_selector_op",
        ":tpu_reshard_variables_op",
        ":transfer_ops",
    ],
)

cc_library(
    name = "tpu_compile_op_common",
    srcs = ["tpu_compile_op_common.cc"],
    hdrs = ["tpu_compile_op_common.h"],
    deps = if_libtpu(
        [":tpu_compilation_metrics"],
        ["//tensorflow/core/tpu/kernels:tpu_compilation_metrics"],
    ) + [
        ":tpu_compilation_cache_entry_unloader",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_metrics_hdrs",
        ":tpu_compile_op_options",
        ":tpu_compile_op_support",
        ":tpu_mesh_state_interface",
        ":tpu_op_consts",
        ":tpu_op_util",
        ":tpu_program_group_interface",
        ":tpu_util",
        ":tpu_fingerprint_lookup",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        ":tpu_util_hdrs",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_op_options",
    srcs = ["tpu_compile_op_options.cc"],
    hdrs = ["tpu_compile_op_options.h"],
)

tf_kernel_library(
    name = "tpu_configuration_ops",
    srcs = ["tpu_configuration_ops.cc"],
    hdrs = ["tpu_configuration_ops.h"],
    deps = if_libtpu(
        [":tpu_util"],
        ["//tensorflow/core/tpu/kernels:tpu_util"],
    ) + [
        ":tpu_compilation_cache_factory",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_local_lookup",
        ":tpu_compilation_cache_lookup",
        ":tpu_compilation_cache_rpc_lookup",
        ":tpu_execute_op_options",
        ":tpu_mesh_state_interface",
        ":tpu_embedding_engine_state_interface",
        ":tpu_op_consts",
        ":tpu_pod_state",
        ":tpu_fingerprint_lookup",
        "@com_google_absl//absl/cleanup",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/tpu:proto_helper",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "tpu_embedding_configuration_ops",
    srcs = ["tpu_embedding_configuration_ops.cc"],
    deps = if_libtpu(
        [":tpu_util"],
        ["//tensorflow/core/tpu/kernels:tpu_util"],
    ) + [
        ":tpu_compilation_cache_factory",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_local_lookup",
        ":tpu_compilation_cache_lookup",
        ":tpu_compilation_cache_rpc_lookup",
        ":tpu_execute_op_options",
        ":tpu_mesh_state_interface",
        ":tpu_embedding_engine_state_interface",
        ":tpu_op_consts",
        ":tpu_pod_state",
        ":tpu_fingerprint_lookup",
        "@com_google_absl//absl/cleanup",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/stream_executor/tpu:status_helper",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/tpu:proto_helper",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_load_retrieve_ops",
    srcs = ["tpu_embedding_load_retrieve_ops.cc"],
    hdrs = ["tpu_embedding_load_retrieve_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_embedding_configuration_proto_rewrite",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_shape_util",
        "//tensorflow/stream_executor/tpu:c_api_conversions",
        "//tensorflow/stream_executor/tpu:c_api_decl",
        "//tensorflow/stream_executor/tpu:status_helper",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_enqueue_ops",
    srcs = ["tpu_embedding_enqueue_ops.cc"],
    hdrs = ["tpu_embedding_enqueue_ops.h"],
    deps = [
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor/tpu:c_api_decl",
        "//tensorflow/stream_executor/tpu:status_helper",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "tpu_executable_info_proto",
    srcs = ["tpu_executable_info.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/core:protos_all",
    ],
)

tf_proto_library(
    name = "tpu_compile_proto",
    srcs = ["tpu_compile.proto"],
    cc_api_version = 2,
    protodeps = [
        ":tpu_executable_info_proto",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/core:protos_all",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto",
    ],
)

cc_library(
    name = "tpu_compilation_cache_factory",
    srcs = ["tpu_compilation_cache_factory.cc"],
    hdrs = ["tpu_compilation_cache_factory.h"],
    deps = [
        ":tpu_compilation_cache_external",
        ":tpu_compilation_cache_interface",
        ":tpu_op_consts",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "tpu_compilation_cache_key",
    srcs = [],
    hdrs = [
        "tpu_compilation_cache_key.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tpu_compile_op_support",
    srcs = ["tpu_compile_op_support.cc"],
    hdrs = ["tpu_compile_op_support.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_compile_proto_cc",
        ":tpu_executable_info_proto_cc",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/stream_executor/tpu:proto_helper",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "tpu_compilation_cache_entry",
    hdrs = [
        "tpu_compilation_cache_entry.h",
    ],
    deps = [
        ":tpu_executable_info_proto_cc",
        ":tpu_program_group_interface",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core/lib/core:refcount",
    ],
)

cc_library(
    name = "tpu_compilation_cache_lookup",
    hdrs = [
        "tpu_compilation_cache_lookup.h",
    ],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_interface",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "tpu_compilation_cache_local_lookup",
    srcs = ["tpu_compilation_cache_local_lookup.cc"],
    hdrs = ["tpu_compilation_cache_local_lookup.h"],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_external",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "tpu_embedding_engine_state_interface",
    srcs = [],
    hdrs = ["tpu_embedding_engine_state_interface.h"],
    deps = [
        "//tensorflow/compiler/xla/service",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_mesh_state_interface",
    srcs = [],
    hdrs = ["tpu_mesh_state_interface.h"],
    deps = [
        "//tensorflow/compiler/xla/service",
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "compiled_subgraph",
    hdrs = ["compiled_subgraph.h"],
    deps = [
        ":tpu_program_group_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:refcount",
    ],
)

cc_library(
    name = "tpu_program_group_interface",
    hdrs = ["tpu_program_group_interface.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_executable_info_proto_cc",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_program_group",
    srcs = ["tpu_program_group.cc"],
    hdrs = ["tpu_program_group.h"],
    deps = [
        ":tpu_compile_op_common",
        ":tpu_compile_op_support",
        ":tpu_compile_proto_cc",
        ":tpu_executable_info_proto_cc",
        ":tpu_mesh_state_interface",
        ":tpu_program_group_interface",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:proto_helper",
        "//tensorflow/stream_executor/tpu:status_helper",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tpu_compilation_cache_interface",
    srcs = ["tpu_compilation_cache_interface.cc"],
    hdrs = ["tpu_compilation_cache_interface.h"],
    deps = if_libtpu(
        [":tpu_compilation_metrics"],
        ["//tensorflow/core/tpu/kernels:tpu_compilation_metrics"],
    ) + [
        ":compiled_subgraph",
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_key",
        ":tpu_compilation_metrics_hdrs",
        ":tpu_util",
        ":tpu_util_hdrs",
        ":trace_util_hdrs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_api",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compilation_cache_external",
    srcs = ["tpu_compilation_cache_external.cc"],
    hdrs = [
        "tpu_compilation_cache_external.h",
    ],
    deps = [
        ":compiled_subgraph",
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_key",
        ":tpu_compilation_metrics",  # buildcleaner: keep
        ":tpu_compilation_metrics_hdrs",
        ":tpu_compile_op_support",
        ":tpu_mesh_state_interface",
        ":tpu_op_consts",
        ":tpu_program_group",
        ":tpu_util",
        ":trace_util_hdrs",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_compilation_metrics_hdrs",
    hdrs = ["tpu_compilation_metrics.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_compilation_metrics",
    srcs = ["tpu_compilation_metrics.cc"],
    copts = tf_copts(),
    deps = [
        ":tpu_compilation_metrics_hdrs",
    ],
)

cc_library(
    name = "trace_util_hdrs",
    hdrs = ["trace_util.h"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_util_hdrs",
    hdrs = ["tpu_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_program_group_interface",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:protos_all_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_op_util",
    srcs = ["tpu_op_util.cc"],
    hdrs = ["tpu_op_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_mesh_state_interface",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_util",
    srcs = ["tpu_util.cc"],
    hdrs = ["tpu_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

# An alias for
cc_library(
    name = "tpu_compilation_cache_cc_proto",
    deps = [":tpu_compilation_cache_proto_cc"],
)

cc_library(
    name = "tpu_compilation_cache_rpc_support_hdrs",
    hdrs = ["tpu_compilation_cache_rpc_support.h"],
    copts = tf_copts(),
    deps = if_libtpu(
        [":tpu_compilation_cache_proto_cc"],
        ["//tensorflow/core/tpu/kernels:tpu_compilation_cache_cc_proto"],
    ) + [
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        ":tpu_program_group_interface",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/platform:status",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_compilation_cache_rpc_support",
    srcs = ["tpu_compilation_cache_rpc_support.cc"],
    copts = tf_copts(),
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_proto_cc",
        ":tpu_compilation_cache_rpc_support_hdrs",
        ":tpu_program_group",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:proto_helper",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "tpu_compilation_cache_rpc_lookup",
    srcs = ["tpu_compilation_cache_rpc_lookup.cc"],
    hdrs = ["tpu_compilation_cache_rpc_lookup.h"],
    copts = tf_copts(),
    deps = if_libtpu(
        [":tpu_compilation_cache_rpc_support"],
        ["//tensorflow/core/tpu/kernels:tpu_compilation_cache_rpc_support"],
    ) + [
        ":tpu_compilation_cache_grpc",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_rpc_support_hdrs",
        ":tpu_program_group_interface",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_cc_dependencies(),
)

tf_proto_library(
    name = "tpu_compilation_cache_proto",
    srcs = ["tpu_compilation_cache.proto"],
    has_services = True,
    cc_api_version = 2,
    create_java_proto = False,
    protodeps = [
        ":tpu_compilation_cache_common_proto",
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto",
    ],
)

tf_proto_library(
    name = "tpu_compilation_cache_common_proto",
    srcs = ["tpu_compilation_cache_common.proto"],
    cc_api_version = 2,
    create_java_proto = False,
)

cc_library(
    name = "tpu_compilation_cache_grpc",
    srcs = ["tpu_compilation_cache_grpc.cc"],
    hdrs = ["tpu_compilation_cache_grpc.h"],
    copts = tf_copts(),
    deps = if_libtpu(
        [":tpu_compilation_cache_proto_cc"],
        ["//tensorflow/core/tpu/kernels:tpu_compilation_cache_cc_proto"],
    ) + [
        ":tpu_compilation_cache_common_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_compilation_cache_service",
    srcs = ["tpu_compilation_cache_service.cc"],
    hdrs = ["tpu_compilation_cache_service.h"],
    copts = tf_copts(),
    deps = if_libtpu(
        [
            ":tpu_compilation_cache_rpc_support",
            ":tpu_compilation_cache_proto_cc",
        ],
        [
            "//tensorflow/core/tpu/kernels:tpu_compilation_cache_rpc_support",
            "//tensorflow/core/tpu/kernels:tpu_compilation_cache_cc_proto",
        ],
    ) + [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_grpc",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_rpc_support_hdrs",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/lib/core:threadpool",
        "//tensorflow/core/platform:coding",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_compile_op_hdrs",
    hdrs = ["tpu_compile_op.h"],
    deps = [
        ":tpu_compile_op_common",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tpu_compilation_cache_entry_unloader",
    hdrs = ["tpu_compilation_cache_entry_unloader.h"],
    deps = [
        ":tpu_compilation_cache_interface",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_op_consts",
    srcs = ["tpu_op_consts.cc"],
    hdrs = ["tpu_op_consts.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "tpu_compile_op_impl",
    srcs = ["tpu_compile_op_impl.cc"],
    hdrs = ["tpu_compile_op_impl.h"],
    copts = tf_copts(),
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_compile_op_common",
        ":tpu_compile_op_support",
        ":tpu_compile_proto_cc",
        ":tpu_program_group",
        ":tpu_program_group_interface",
        ":tpu_util",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_hdrs",
        "@com_google_absl//absl/types:variant",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_op_lib",
    srcs = ["tpu_compile_op.cc"],
    deps = [
        ":tpu_compile_op_hdrs",
        ":tpu_compile_op_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_compile_op",
    deps = [
        ":tpu_compile_op_hdrs",
        ":tpu_compile_op_impl",
        ":tpu_compile_op_lib",
        ":tpu_compile_op_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_execute_op",
    srcs = ["tpu_execute_op.cc"],
    hdrs = ["tpu_execute_op.h"],
    deps = [
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_external",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_local_lookup",
        ":tpu_compilation_cache_lookup",
        ":tpu_executable_info_proto_cc",
        ":tpu_op_consts",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_execute_op_options",
    srcs = ["tpu_execute_op_options.cc"],
    hdrs = ["tpu_execute_op_options.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "cross_replica_ops",
    srcs = ["cross_replica_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "topk_ops",
    srcs = ["topk_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/numeric:bits",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_ops",
    srcs = ["tpu_embedding_ops.cc"],
    deps = [
        ":tpu_mesh_state_interface",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/lib:scatter",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:side_effect_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu/kernels:cross_replica_ops",
        "//tensorflow/stream_executor/tpu:c_api_conversions",
        "//tensorflow/stream_executor/tpu:c_api_decl",
        "//tensorflow/stream_executor/tpu:status_helper",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "host_compute_ops",
    srcs = ["host_compute_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/tpu:tpu_defs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "infeed_ops",
    srcs = ["infeed_ops.cc"],
    hdrs = ["infeed_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_ops",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/kernels:transpose_functor",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor/tpu:c_api_conversions",
        "//tensorflow/stream_executor/tpu:tpu_transfer_manager_base",
        "//tensorflow/stream_executor/tpu:tpu_transfer_manager_interface",
    ],
    alwayslink = True,
)

cc_library(
    name = "transfer_ops",
    srcs = ["transfer_ops.cc"],
    hdrs = ["transfer_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
        "//tensorflow/stream_executor/tpu:tpu_transfer_manager_interface",
    ],
    alwayslink = True,
)

cc_library(
    name = "outfeed_ops",
    srcs = ["outfeed_ops.cc"],
    hdrs = ["outfeed_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":transfer_ops",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/stream_executor:multi_platform_manager",
    ],
    alwayslink = True,
)

cc_library(
    name = "image_resize_ops",
    srcs = ["image_resize_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "replication_ops",
    srcs = ["replication_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_defs",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_handle_to_key_op",
    srcs = ["tpu_handle_to_key_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_compilation_cache_interface",
        ":tpu_op_consts",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_configuration",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_pod_state",
    srcs = ["tpu_pod_state.cc"],
    hdrs = ["tpu_pod_state.h"],
    copts = tf_copts(),
    deps = if_libtpu(
        [":tpu_util"],
        ["//tensorflow/core/tpu/kernels:tpu_util"],
    ) + [
        ":tpu_compilation_cache_service",
        "@com_google_absl//absl/cleanup",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tpu_reshard_variables_op",
    srcs = ["tpu_reshard_variables_op.cc"],
    hdrs = ["tpu_reshard_variables_op.h"],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_lookup",
        ":tpu_op_consts",
        ":tpu_program_group",
        ":tpu_reshard_variables_op_util",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor/tpu:tpu_executor_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_interface",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_reshard_variables_op_util",
    srcs = ["tpu_reshard_variables_op_util.cc"],
    hdrs = ["tpu_reshard_variables_op_util.h"],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        ":tpu_op_consts",
        ":tpu_program_group",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor/tpu:tpu_executor_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_interface",
        "//tensorflow/stream_executor/tpu:tpu_node_context",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_ordinal_selector_op",
    srcs = ["tpu_ordinal_selector_op.cc"],
    deps = [
        ":tpu_ordinal_selector",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_ordinal_selector_interface",
    hdrs = ["tpu_ordinal_selector_interface.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tpu_ordinal_selector",
    hdrs = ["tpu_ordinal_selector.h"],
    deps = [
        ":tpu_ordinal_selector_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_fingerprint_lookup",
    srcs = ["tpu_fingerprint_lookup.cc"],
    hdrs = ["tpu_fingerprint_lookup.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_functional_ops",
    srcs = ["tpu_functional_ops.cc"],
    hdrs = ["tpu_functional_ops.h"],
    deps = [
        ":tpu_compile_op_support",
        ":tpu_op_util",
        ":tpu_ordinal_selector",
        ":tpu_util_hdrs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//third_party/eigen3",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/common_runtime:placer",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_fingerprint_lookup",
        "//tensorflow/core/tpu/kernels:tpu_op_consts",
    ] + if_static(["//tensorflow/core/common_runtime:rendezvous_mgr"]),
    alwayslink = 1,
)

cc_library(
    name = "sharding_util_ops",
    srcs = ["sharding_util_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/framework:kernel_def_proto_cc",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "sharding_util_ops_test",
    srcs = ["sharding_util_ops_test.cc"],
    deps = [
        ":sharding_util_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
