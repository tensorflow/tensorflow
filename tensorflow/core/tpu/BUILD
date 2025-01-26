# Description: Utilities for TPU Operations

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_libtpu",
    "if_windows",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_embedding_configuration_utils",
    srcs = ["tpu_embedding_configuration_utils.cc"],
    hdrs = ["tpu_embedding_configuration_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tpu_embedding_errors",
    srcs = ["tpu_embedding_errors.cc"],
    hdrs = ["tpu_embedding_errors.h"],
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

tf_cc_test(
    name = "tpu_embedding_errors_test",
    srcs = ["tpu_embedding_errors_test.cc"],
    deps = [
        ":tpu_embedding_errors",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "tpu_embedding_optimization_parameters_utils",
    srcs = ["tpu_embedding_optimization_parameters_utils.cc"],
    hdrs = ["tpu_embedding_optimization_parameters_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "tpu_embedding_output_layout_utils",
    srcs = ["tpu_embedding_output_layout_utils.cc"],
    hdrs = ["tpu_embedding_output_layout_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
    ],
)

cc_library(
    name = "tpu_embedding_spmd_sharding_utils",
    srcs = ["tpu_embedding_spmd_sharding_utils.cc"],
    hdrs = ["tpu_embedding_spmd_sharding_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
)

cc_library(
    name = "tpu_embedding_configuration_proto_rewrite",
    srcs = ["tpu_embedding_configuration_proto_rewrite.cc"],
    hdrs = ["tpu_embedding_configuration_proto_rewrite.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "tpu_embedding_configuration_proto_rewrite_test",
    srcs = ["tpu_embedding_configuration_proto_rewrite_test.cc"],
    deps = [
        ":tpu_embedding_configuration_proto_rewrite",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "tpu_node_device_util",
    srcs = ["tpu_node_device_util.cc"],
    hdrs = ["tpu_node_device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tpu_compile_interface",
    srcs = ["tpu_compile_interface.cc"],
    hdrs = ["tpu_compile_interface.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_defs",
    srcs = ["tpu_defs.cc"],
    hdrs = ["tpu_defs.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:protos_all_cc"],
)

cc_library(
    name = "tpu_configuration",
    srcs = ["tpu_configuration.cc"],
    hdrs = ["tpu_configuration.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "tpu_init_mode",
    srcs = ["tpu_init_mode.cc"],
    hdrs = ["tpu_init_mode.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tpu_api_dlsym_initializer",
    srcs = if_windows(
        ["tpu_api_dlsym_initializer_windows.cc"],
        otherwise = ["tpu_api_dlsym_initializer.cc"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla/stream_executor/tpu:libtftpu_header",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_api_dlsym_set_fn",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_base",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_initialize_util",
        "@local_xla//xla/stream_executor/tpu:tpu_library_init_fns",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
    # Always link this in, because even if we don't use it directly we want its
    # static initializers to dynamically load API symbols exported from libtpu.so
    alwayslink = True,
)

cc_library(
    name = "virtual_device",
    srcs = ["virtual_device.cc"],
    hdrs = ["virtual_device.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tpu_compile",
    srcs = ["tpu_compile.cc"],
    hdrs = ["tpu_compile.h"],
    deps = [
        ":tpu_defs",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_resource",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/hlo/ir:hlo",
    ],
)

cc_library(
    name = "tpu_execute",
    srcs = ["tpu_execute.cc"],
    hdrs = ["tpu_execute.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:outside_compilation_params",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu/kernels:tpu_executable_info_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla:shape_layout",
        "@local_xla//xla:shape_tree",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:backend",
        "@local_xla//xla/service:computation_layout",
        "@local_xla//xla/service:computation_placer",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:maybe_owning_device_memory",
        "@local_xla//xla/service:transfer_manager",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:stream",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:proto_helper",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
        "@local_xla//xla/stream_executor/tpu:tpu_op_executable",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
    ],
)

cc_library(
    name = "tpu_runtime",
    srcs = [],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api_dlsym_initializer",
        "//tensorflow/core/tpu/ops",
        "@local_xla//xla/stream_executor/tpu:tpu_on_demand_compiler",
    ],
)

cc_library(
    name = "tpu_fingerprint_utils",
    srcs = ["tpu_fingerprint_utils.cc"],
    hdrs = ["tpu_fingerprint_utils.h"],
    deps = [
        ":tpu_compile_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/core/platform:status",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "tpu_model_server_initializer",
    srcs = ["tpu_model_server_initializer.cc"],
    hdrs = ["tpu_model_server_initializer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@local_xla//xla/stream_executor/tpu:libtftpu_header",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_api_dlsym_set_fn",
        "@local_xla//xla/stream_executor/tpu:tpu_executor",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_init_fns",
        "@local_xla//xla/stream_executor/tpu:tpu_initialize_util",
        "@local_xla//xla/stream_executor/tpu:tpu_library_init_fns",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_global_init",
    srcs = ["tpu_global_init.cc"],
    hdrs = ["tpu_global_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_defs",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:tpu_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu/graph_rewrite:distributed_tpu_configuration_rewrite_pass",
        "//tensorflow/core/tpu/graph_rewrite:distributed_tpu_rewrite_helpers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ] + if_libtpu([
        "//tensorflow/compiler/jit:xla_tpu_jit",
        "//tensorflow/compiler/jit:xla_tpu_device",
    ]),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = ["//tensorflow:tensorflow_bzl"],
)
