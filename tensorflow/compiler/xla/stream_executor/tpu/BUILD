# Description: StreamExecutor Interface for TPUs

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//learning/brain/experimental/dtensor:__subpackages__",
        "//learning/brain/google/xla/kernels:__subpackages__",
        "//learning/brain/tfrc/executor:__subpackages__",
        "//learning/brain/tfrc/runtime/tpu_driver:__subpackages__",
        "//tensorflow/compiler/jit:__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
        "//tensorflow/core/profiler/backends/tpu:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "c_api_decl",
    hdrs = [
        "c_api_decl.h",
        "c_api_defn.h",
    ],
    deps = [
        "//tensorflow/c:tf_attrtype",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/core/tpu:libtftpu_header",
    ],
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        "//tensorflow/c:tf_attrtype",
        "//tensorflow/c:tf_status",
        "//tensorflow/core/tpu:libtftpu_header",
    ],
)

cc_library(
    name = "c_api_conversions",
    srcs = ["c_api_conversions.cc"],
    hdrs = ["c_api_conversions.h"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/core/tpu:tpu_api",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "noncopyable_buffer",
    hdrs = ["noncopyable_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_api",
    ],
)

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = [
        ":c_api_decl",
        "//tensorflow/tsl/platform:logging",
    ],
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
        "tpu_timer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":status_helper",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_interface",
        ":tpu_stream_interface",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_platform_hdr",
    hdrs = ["tpu_platform.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
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
        "tpu_timer.h",
    ],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_topology_external",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_platform_id",
    srcs = ["tpu_platform_id.cc"],
    hdrs = ["tpu_platform_id.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor:platform"],
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
        "tpu_timer.h",
    ],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_id",
        ":tpu_platform_interface",
        ":tpu_stream_interface",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:casts",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_node_context",
    srcs = ["tpu_node_context.cc"],
    hdrs = ["tpu_node_context.h"],
    deps = [
        ":status_helper",
        ":tpu_platform_interface",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:macros",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/xla/service:transfer_manager",
        "@com_google_absl//absl/cleanup",
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
        "//tensorflow/compiler/xla/service:transfer_manager",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_transfer_manager_base",
    srcs = ["tpu_transfer_manager.cc"],
    hdrs = ["tpu_transfer_manager.h"],
    deps = [
        ":c_api_conversions",
        ":noncopyable_buffer",
        ":proto_helper",
        ":status_helper",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_id",
        ":tpu_transfer_manager_interface",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/core/tpu:tpu_api",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "tpu_op_executable",
    srcs = ["tpu_op_executable.cc"],
    hdrs = ["tpu_op_executable.h"],
    deps = [
        ":c_api_conversions",
        ":proto_helper",
        ":status_helper",
        ":tpu_executable_interface",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/tsl/platform:casts",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_executor_interface",
    hdrs = ["tpu_executor_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_platform_interface",
        ":tpu_topology_external",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
    ],
)

cc_library(
    name = "tpu_stream_interface",
    hdrs = ["tpu_stream_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "tpu_executable_interface",
    srcs = ["tpu_executable_interface.cc"],
    hdrs = ["tpu_executable_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_profile_printer_data_cc",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_executable",
    srcs = ["tpu_executable.cc"],
    hdrs = ["tpu_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_conversions",
        ":status_helper",
        ":tpu_executable_interface",
        ":tpu_executor",
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/core/tpu:tpu_executor_api",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "tpu_topology_external",
    srcs = ["tpu_topology.cc"],
    hdrs = ["tpu_topology.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:types",
    ],
)
