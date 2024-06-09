# Description: StreamExecutor Interface for TPUs

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/experimental/dtensor:__subpackages__",
        "//learning/brain/google/xla/kernels:__subpackages__",
        "//learning/brain/research/pjrt:__subpackages__",
        "//learning/brain/tfrc/executor:__subpackages__",
        "//learning/brain/tfrt/tpu_plugin:__subpackages__",
        "//tensorflow/compiler/jit:__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//xla:__subpackages__",
        "//xla/backends/profiler/tpu:__subpackages__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "c_api_decl",
    hdrs = [
        "c_api_decl.h",
        "c_api_defn.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        "//xla/stream_executor",
    ],
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
    ],
)

cc_library(
    name = "c_api_conversions",
    srcs = ["c_api_conversions.cc"],
    hdrs = ["c_api_conversions.h"],
    deps = [
        ":c_api_decl",
        ":proto_helper",
        ":tpu_api",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        ":tpu_ops_c_api_hdrs",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "c_api_conversions_test",
    srcs = ["c_api_conversions_test.cc"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "libtftpu_header",
    hdrs = ["libtftpu.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "noncopyable_buffer",
    hdrs = ["noncopyable_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tsl_status_helper",
    hdrs = ["tsl_status_helper.h"],
    deps = [
        ":c_api_decl",
        "//xla/tsl/c:tsl_status",
        "//xla/tsl/c:tsl_status_helper",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = [
        ":c_api_decl",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_api_dlsym_set_fn",
    hdrs = ["tpu_api_dlsym_set_fn.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_ops_c_api_hdrs",
    srcs = [],
    hdrs = [
        "tpu_ops_c_api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_executor",
    srcs = [
        "tpu_platform_registration.cc",
    ],
    hdrs = [
        "tpu_executor.h",
        "tpu_platform.h",
        "tpu_stream.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":status_helper",
        ":tpu_executor_api",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_interface",
        ":tpu_stream_interface",
        ":tpu_topology_external",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_executor_init_fns",
    hdrs = ["tpu_executor_init_fns.inc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_platform_hdr",
    hdrs = ["tpu_platform.h"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_interface",
        ":tpu_topology_external",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_executor_hdrs",
    hdrs = [
        "tpu_executor.h",
        "tpu_executor_interface.h",
        "tpu_platform.h",
        "tpu_platform_interface.h",
        "tpu_stream.h",
        "tpu_stream_interface.h",
    ],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":status_helper",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        ":tpu_topology_external",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "tpu_platform_id",
    srcs = ["tpu_platform_id.cc"],
    hdrs = ["tpu_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "tpu_executor_base",
    srcs = [
        "tpu_executor.cc",
        "tpu_platform.cc",
    ],
    hdrs = [
        "tpu_event.h",
        "tpu_executor.h",
        "tpu_platform.h",
        "tpu_stream.h",
    ],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":status_helper",
        ":tpu_api",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_id",
        ":tpu_platform_interface",
        ":tpu_stream_interface",
        ":tpu_topology_external",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/tsl/c:tsl_status_internal",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "tpu_library_init_fns",
    hdrs = ["tpu_library_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_executor_init_fns",
        ":tpu_ops_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_profiler_init_fns",
    hdrs = ["tpu_profiler_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api",
        ":tpu_profiler_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_node_context",
    srcs = ["tpu_node_context.cc"],
    hdrs = ["tpu_node_context.h"],
    deps = [
        ":status_helper",
        ":tpu_api",
        ":tpu_ops_c_api_hdrs",
        ":tpu_platform_interface",
        "//xla/service",
        "//xla/service:backend",
        "//xla/service:stream_pool",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:macros",
    ],
)

cc_library(
    name = "tpu_api",
    srcs = ["tpu_api.cc"],
    hdrs = ["tpu_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_executor_api",
        ":tpu_ops_c_api_hdrs",
        ":tpu_profiler_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_transfer_manager_interface",
    srcs = ["tpu_transfer_manager_interface.cc"],
    hdrs = ["tpu_transfer_manager_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":noncopyable_buffer",
        ":tpu_platform_interface",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tpu_transfer_manager",
    srcs = ["tpu_transfer_manager_registration.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_executor",
        ":tpu_platform_id",
        ":tpu_transfer_manager_base",
        "//xla/service:transfer_manager",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_transfer_manager_base",
    srcs = ["tpu_transfer_manager.cc"],
    hdrs = ["tpu_transfer_manager.h"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":noncopyable_buffer",
        ":status_helper",
        ":tpu_executor_api",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_id",
        ":tpu_transfer_manager_interface",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_op_executable",
    srcs = ["tpu_op_executable.cc"],
    hdrs = ["tpu_op_executable.h"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":proto_helper",
        ":status_helper",
        ":tpu_api",
        ":tpu_executable_interface",
        ":tpu_executor_base",
        ":tpu_ops_c_api_hdrs",
        ":tpu_platform_interface",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
    ],
)

cc_library(
    name = "tpu_platform_interface",
    srcs = ["tpu_platform_interface.cc"],
    hdrs = ["tpu_platform_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":tpu_topology_external",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "tpu_executor_interface",
    hdrs = ["tpu_executor_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_platform_interface",
        ":tpu_topology_external",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tpu_initialize_util",
    srcs = ["tpu_initialize_util.cc"],
    hdrs = [
        "tpu_initialize_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_on_demand_compiler",
    srcs = ["tpu_on_demand_compiler.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":proto_helper",
        ":status_helper",
        ":tpu_executable",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_hdrs",
        ":tpu_platform_id",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_stream_interface",
    hdrs = ["tpu_stream_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tpu_executable_interface",
    srcs = ["tpu_executable_interface.cc"],
    hdrs = ["tpu_executable_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_executable",
    srcs = ["tpu_executable.cc"],
    hdrs = ["tpu_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":proto_helper",
        ":status_helper",
        ":tpu_executable_interface",
        ":tpu_executor",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_executor_api",
    srcs = ["tpu_executor_api.cc"],
    hdrs = ["tpu_executor_api.h"],
    deps = [
        ":libtftpu_header",
        ":tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_profiler_c_api_hdrs",
    hdrs = ["tpu_profiler_c_api.h"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
    ],
)

cc_library(
    name = "tpu_topology_external",
    srcs = ["tpu_topology.cc"],
    hdrs = ["tpu_topology.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_api",
        "@local_tsl//tsl/platform:types",
    ],
)
