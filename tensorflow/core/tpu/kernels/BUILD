# TPU Kernel Implementations
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library_cc",
)

package(
    default_visibility = [
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/stream_executor/tpu:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tpu_compile_op_common",
    srcs = ["tpu_compile_op_common.cc"],
    hdrs = ["tpu_compile_op_common.h"],
    deps = [
        ":tpu_compile_op_support",
        ":tpu_mesh_state_interface",
        ":tpu_util",
        ":tpu_util_hdrs",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        # "//tensorflow/core/protobuf/tpu:compilation_result_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
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

cc_library(
    name = "tpu_configuration_ops",
    srcs = ["tpu_configuration_ops.cc"],
    hdrs = ["tpu_configuration_ops.h"],
    deps = [
        ":tpu_mesh_state_interface",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tpu:tpu_config_c_api",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_library_loader",
        "//tensorflow/stream_executor/tpu:proto_helper",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_c_api_hdrs",
    hdrs = ["tpu_compile_c_api.h"],
    deps = [
        ":tpu_mesh_state_c_api",
        ":tpu_ops_common_c_api_hdrs",
        ":tpu_program_c_api_hdrs",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/stream_executor/tpu:proto_helper",
    ],
)

tf_proto_library_cc(
    name = "tpu_executable_info_proto",
    srcs = ["tpu_executable_info.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/core:protos_all",
    ],
)

tf_proto_library_cc(
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
    name = "tpu_compilation_cache_key",
    srcs = [],
    hdrs = [
        "tpu_compilation_cache_key.h",
    ],
    deps = ["@com_google_absl//absl/types:optional"],
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
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/stream_executor/tpu:proto_helper",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_compilation_cache_entry",
    srcs = ["tpu_compilation_cache_entry.cc"],
    hdrs = [
        "tpu_compilation_cache_entry.h",
    ],
    deps = [
        ":tpu_executable_info_proto_cc",
        ":tpu_program_group",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/platform:casts",
    ],
)

cc_library(
    name = "tpu_compilation_cache_lookup",
    srcs = ["tpu_compilation_cache_lookup.cc"],
    hdrs = [
        "tpu_compilation_cache_lookup.h",
    ],
    deps = [
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_external",
        ":tpu_compilation_cache_proto_cc",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "tpu_mesh_state_c_api",
    hdrs = ["tpu_mesh_state_c_api.h"],
)

cc_library(
    name = "tpu_mesh_state_interface",
    srcs = [],
    hdrs = ["tpu_mesh_state_interface.h"],
    deps = [
        ":tpu_compile_c_api_hdrs",
        ":tpu_mesh_state_c_api",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_config_c_api",
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
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "tpu_program_group",
    srcs = ["tpu_program_group.cc"],
    hdrs = ["tpu_program_group.h"],
    deps = [
        ":tpu_compile_c_api_hdrs",
        ":tpu_compile_op_common",
        ":tpu_compile_op_support",
        ":tpu_compile_proto_cc",
        ":tpu_executable_info_proto_cc",
        ":tpu_program_c_api_hdrs",
        ":tpu_program_group_interface",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/stream_executor/tpu:proto_helper",
        "//tensorflow/stream_executor/tpu:status_helper",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tpu_compilation_cache_external",
    srcs = ["tpu_compilation_cache_external.cc"],
    hdrs = [
        "tpu_compilation_cache_external.h",
    ],
    deps = [
        ":compiled_subgraph",
        ":tpu_compilation_cache_entry",
        ":tpu_compilation_cache_key",
        ":tpu_compilation_cache_metrics",  # buildcleaner: keep
        ":tpu_compilation_cache_metrics_hdrs",
        ":tpu_compilation_cache_proto_cc",
        ":tpu_compile_c_api_hdrs",
        ":tpu_compile_op_support",
        ":tpu_mesh_state_interface",
        ":tpu_program_group",
        ":tpu_util",
        ":trace_util_hdrs",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_compilation_cache_metrics_hdrs",
    hdrs = ["tpu_compilation_cache_metrics.h"],
    deps = [
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_compilation_cache_metrics",
    srcs = ["tpu_compilation_cache_metrics.cc"],
    deps = [
        ":tpu_compilation_cache_metrics_hdrs",
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
    srcs = [],
    hdrs = ["tpu_util.h"],
    deps = [
        ":tpu_compilation_cache_key",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_util_c_api_hdrs",
    hdrs = ["tpu_util_c_api.h"],
    deps = [
        "//tensorflow/stream_executor/tpu:proto_helper",
    ],
)

cc_library(
    name = "tpu_ops_common_c_api_hdrs",
    hdrs = ["tpu_ops_common_c_api.h"],
)

cc_library(
    name = "tpu_program_c_api_hdrs",
    hdrs = ["tpu_program_c_api.h"],
    deps = [
        ":tpu_ops_common_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:proto_helper",
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
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_proto_library_cc(
    name = "tpu_compilation_cache_proto",
    srcs = ["tpu_compilation_cache.proto"],
    cc_api_version = 2,
)
