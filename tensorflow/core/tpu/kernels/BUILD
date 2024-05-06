# TPU Kernel Implementations

# Placeholder: load py_proto_library
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
    "if_libtpu",
    "if_oss",
    "tf_cc_test",
    "tf_copts",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "tf_grpc_cc_dependencies",
    "tf_kernel_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/tpu:build_defs.bzl",
    "if_libtpu_tf_status",
    "if_libtpu_tf_tensor",
)

# Config setting to enable go/libtpu support.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [] + if_google([
        "//learning/brain/google/xla:friends",
    ]),
    packages = [
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/...",
        "//tensorflow/core/tfrt/ifrt/...",
        "//tensorflow/core/tpu/...",
        "//tensorflow/dtensor/...",
        "//third_party/py/jax_tpu_embedding/...",
    ] + if_google([
        "//learning/brain/google/xla/...",
    ]),
)

tf_kernel_library(
    name = "kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":cross_replica_ops",
        ":global_iter_id_op",
        ":host_compute_ops",
        ":image_resize_ops",
        ":infeed_ops",
        ":outfeed_ops",
        ":replication_ops",
        ":sharding_util_ops",
        ":sparse_core_preprocess_ops",
        ":sparse_core_xla_ops",
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
        ":tpu_compilation_cache_key",
        ":tpu_compilation_metrics_hdrs",
        ":tpu_compile_op_options",
        ":tpu_compile_op_support",
        ":tpu_fingerprint_lookup",
        ":tpu_mesh_state_interface",
        ":tpu_op_consts",
        ":tpu_op_util",
        ":tpu_program_group_interface",
        ":tpu_util",
        ":tpu_util_hdrs",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_op_options",
    srcs = ["tpu_compile_op_options.cc"],
    hdrs = ["tpu_compile_op_options.h"],
)

cc_library(
    name = "sparse_core_preprocess_ops",
    srcs = ["sparse_core_preprocess_ops.cc"],
    hdrs = ["sparse_core_preprocess_ops.h"],
    deps = [
        ":sparse_core_ops_stats_handler",
        ":sparse_core_ops_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_highway//:hwy",
        "@com_google_highway//hwy/contrib/sort:vqsort",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_xla//xla:util",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "sparse_core_xla_ops",
    srcs = ["sparse_core_xla_ops.cc"],
    hdrs = ["sparse_core_xla_ops.h"],
    deps = [
        ":sparse_core_ops_utils",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/client/lib:slicing",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_core_ops_stats_handler",
    hdrs = ["sparse_core_ops_stats_handler.h"],
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
        ":tpu_embedding_engine_state_interface",
        ":tpu_execute_op_options",
        ":tpu_fingerprint_lookup",
        ":tpu_mesh_state_interface",
        ":tpu_op_consts",
        ":tpu_pod_state",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:tstring",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla:util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ] + if_libtpu_tf_status(),
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
        ":tpu_embedding_engine_state_interface",
        ":tpu_execute_op_options",
        ":tpu_fingerprint_lookup",
        ":tpu_mesh_state_interface",
        ":tpu_op_consts",
        ":tpu_pod_state",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/cleanup",
        "@local_xla//xla:util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ] + if_libtpu_tf_status(),
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
        "//tensorflow/core/tpu:tpu_embedding_configuration_proto_rewrite",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_shape_util",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_enqueue_ops",
    srcs = ["tpu_embedding_enqueue_ops.cc"],
    hdrs = ["tpu_embedding_enqueue_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:tstring",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
    ] + if_libtpu_tf_tensor(),
    alwayslink = 1,
)

tf_proto_library(
    name = "tpu_executable_info_proto",
    srcs = ["tpu_executable_info.proto"],
    cc_api_version = 2,
    protodeps = [
        "@local_xla//xla:xla_data_proto",
        "@local_xla//xla/service:hlo_proto",
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
        "@local_xla//xla:xla_data_proto",
        "@local_xla//xla/service:hlo_proto",
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
    ],
)

cc_library(
    name = "tpu_compilation_cache_key",
    srcs = [],
    hdrs = [
        "tpu_compilation_cache_key.h",
    ],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "tpu_compile_op_support",
    srcs = ["tpu_compile_op_support.cc"],
    hdrs = ["tpu_compile_op_support.h"],
    visibility = [
        ":friends",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":tpu_compile_proto_cc",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:shape_tree",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:computation_layout",
        "@local_xla//xla/service:computation_placer",
        "@local_xla//xla/service:dump",
        "@local_xla//xla/service:hlo_module_config",
    ],
)

cc_library(
    name = "tpu_compilation_cache_entry",
    hdrs = [
        "tpu_compilation_cache_entry.h",
    ],
    deps = [":tpu_program_group_interface"],
)

cc_library(
    name = "tpu_compilation_cache_lookup",
    hdrs = [
        "tpu_compilation_cache_lookup.h",
    ],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_interface",
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "tpu_compilation_cache_local_lookup",
    srcs = ["tpu_compilation_cache_local_lookup.cc"],
    hdrs = ["tpu_compilation_cache_local_lookup.h"],
    deps = [
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_embedding_engine_state_interface",
    srcs = [],
    hdrs = ["tpu_embedding_engine_state_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "@local_xla//xla/service",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_mesh_state_interface",
    srcs = [],
    hdrs = ["tpu_mesh_state_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@local_xla//xla/service",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        ":tpu_executable_info_proto_cc",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/hlo/ir:hlo_module_group",
        "@local_xla//xla/service:computation_placer",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
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
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla:util",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
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
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_key",
        ":tpu_compilation_metrics",  # buildcleaner: keep
        ":tpu_program_group",
        ":tpu_program_group_interface",
        ":tpu_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@local_xla//xla/service",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_compilation_metrics_hdrs",
    hdrs = ["tpu_compilation_metrics.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "tpu_compilation_metrics",
    srcs = ["tpu_compilation_metrics.cc"],
    copts = tf_copts(),
    deps = [
        ":tpu_compilation_metrics_hdrs",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_util_hdrs",
    hdrs = ["trace_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_util_hdrs",
    hdrs = ["tpu_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_program_group_interface",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_xla//xla:statusor",
        "@local_xla//xla/client:compile_only_client",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "tpu_op_util",
    srcs = ["tpu_op_util.cc"],
    hdrs = ["tpu_op_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        ":tpu_mesh_state_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_util",
    srcs = ["tpu_util.cc"],
    hdrs = ["tpu_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla:statusor",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/cleanup",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        ":tpu_compilation_cache_common_proto_cc",
        ":tpu_compilation_cache_grpc",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        ":tpu_compilation_cache_rpc_support_hdrs",
        ":tpu_program_group_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/lib/core:threadpool",
        "//tensorflow/core/platform:coding",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_call",
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
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
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
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/stream_executor/platform",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_op_lib",
    srcs = ["tpu_compile_op.cc"],
    deps = [
        ":tpu_compile_op_common",
        ":tpu_compile_op_hdrs",
        ":tpu_compile_op_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:tstring",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/stream_executor/platform",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
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
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "@local_xla//xla:statusor",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_execute_op",
    srcs = ["tpu_execute_op.cc"],
    hdrs = ["tpu_execute_op.h"],
    deps = [
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_interface",
        ":tpu_compilation_cache_lookup",
        ":tpu_executable_info_proto_cc",
        ":tpu_op_consts",
        ":tpu_program_group",
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:variable_info_util",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_tree",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:backend",
        "@local_xla//xla/service:computation_placer_hdr",
        "@local_xla//xla/service:dump",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/service:maybe_owning_device_memory",
        "@local_xla//xla/service:shaped_buffer",
        "@local_xla//xla/service:transfer_manager",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
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
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:xla_builder",
    ],
    alwayslink = 1,
)

cc_library(
    name = "topk_ops",
    srcs = ["topk_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/numeric:bits",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client/lib:arithmetic",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_ops",
    srcs = ["tpu_embedding_ops.cc"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_spmd_sharding_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/kernels:transpose_functor",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:noncopyable_buffer",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_api",
        "@local_xla//xla/stream_executor/tpu:tpu_transfer_manager_interface",
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
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla:literal",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/tpu:noncopyable_buffer",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_transfer_manager_interface",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_defs",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
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
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client/lib:constants",
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
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/cleanup",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
    ] + if_libtpu_tf_status(),
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
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:variable_info_util",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "@local_xla//xla/service:maybe_owning_device_memory",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
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
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "@local_xla//xla/service:maybe_owning_device_memory",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
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
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_fingerprint_lookup",
    srcs = ["tpu_fingerprint_lookup.cc"],
    hdrs = ["tpu_fingerprint_lookup.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:stringpiece",
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
        ":tpu_fingerprint_lookup",
        ":tpu_op_consts",
        ":tpu_op_util",
        ":tpu_ordinal_selector",
        ":tpu_util_hdrs",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/core:core_cpu",
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
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:array4d",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_topology_external",
    ] + if_static(["//tensorflow/core/common_runtime:rendezvous_mgr"]),
    alwayslink = 1,
)

cc_library(
    name = "sharding_util_ops",
    srcs = ["sharding_util_ops.cc"],
    deps = [
        ":sharding_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sharding_utils",
    srcs = ["sharding_utils.cc"],
    hdrs = ["sharding_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "sharding_utils_test",
    srcs = ["sharding_utils_test.cc"],
    deps = [
        ":sharding_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_kernel_library(
    name = "global_iter_id_op",
    srcs = ["global_iter_id.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:partitioned_function_ops",
        "//tensorflow/core/tpu/ops:sparse_core_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_global_iter_id_op",
    out = "gen_global_iter_id_op.py",
    op_allowlist = [
        "GlobalIterId",
    ],
    visibility = ["//visibility:public"],
    deps = [":global_iter_id_op"],
)

tf_cc_test(
    name = "sharding_util_ops_test",
    srcs = ["sharding_util_ops_test.cc"],
    deps = [
        ":sharding_util_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "sparse_core_ops_utils",
    srcs = ["sparse_core_ops_utils.cc"],
    hdrs = ["sparse_core_ops_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sparse_core_xla_flags_defaults",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "sparse_core_xla_flags_defaults",
    hdrs = ["sparse_core_xla_flags_defaults.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sparse_core_layout",
    srcs = ["sparse_core_layout.cc"],
    hdrs = ["sparse_core_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sparse_core_layout_proto_cc",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

tf_cc_test(
    name = "sparse_core_layout_test",
    srcs = ["sparse_core_layout_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":sparse_core_layout",
        ":sparse_core_layout_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "sparse_core_ops_utils_test",
    srcs = ["sparse_core_ops_utils_test.cc"],
    deps = [
        ":sparse_core_ops_utils",
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "sparse_core_layout_proto",
    srcs = ["sparse_core_layout.proto"],
    has_services = False,
    cc_api_version = 2,
    create_go_proto = False,
    create_java_proto = False,
    create_kotlin_proto = False,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "sparse_core_layout_py_pb2",
#     api_version = 2,
#     deps = [":sparse_core_layout_proto"],
# )
# copybara:uncomment_end
