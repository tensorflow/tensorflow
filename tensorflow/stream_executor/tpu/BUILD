# Description: StreamExecutor Interface for TPUs

package(
    default_visibility = [
        "//learning/brain/tfrc/runtime/tpu_driver:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    deps = [
        "//tensorflow/c:tf_attrtype",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status",
        "//tensorflow/core/tpu/kernels:tpu_ops_common_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_node_context_c_api_hdrs",
    hdrs = ["tpu_node_context_c_api.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "c_api_conversions",
    hdrs = ["c_api_conversions.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/stream_executor:stream",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "tpu_stream",
    hdrs = ["tpu_stream.h"],
    deps = [
        ":c_api_conversions",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_stream_interface",
        "//tensorflow/stream_executor:stream",
    ],
)

cc_library(
    name = "tpu_timer",
    hdrs = ["tpu_timer.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:stream",
    ],
)

cc_library(
    name = "tpu_executor",
    srcs = ["tpu_executor.cc"],
    hdrs = ["tpu_executor.h"],
    deps = [
        ":c_api_conversions",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform",
        ":tpu_platform_interface",
        ":tpu_stream",
        ":tpu_timer",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "tpu_executor_hdrs",
    hdrs = ["tpu_executor.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "tpu_platform_hdrs",
    hdrs = ["tpu_platform.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "tpu_node_context",
    srcs = ["tpu_node_context.cc"],
    hdrs = ["tpu_node_context.h"],
    deps = [
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_node_context_c_api_hdrs",
        ":tpu_platform_interface",
        ":tpu_transfer_manager",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:framework",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tpu_platform",
    srcs = ["tpu_platform.cc"],
    hdrs = ["tpu_platform.h"],
    deps = [
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_hdrs",
        ":tpu_platform_interface",
        "//tensorflow/c:tf_status",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:stream",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_transfer_manager",
    srcs = ["tpu_transfer_manager_registration.cc"],
    deps = [
        ":tpu_platform",
        ":tpu_transfer_manager_base",
        "//tensorflow/compiler/xla/service:transfer_manager",
    ],
)

cc_library(
    name = "tpu_transfer_manager_base",
    srcs = ["tpu_transfer_manager.cc"],
    hdrs = ["tpu_transfer_manager.h"],
    deps = [
        ":c_api_conversions",
        ":proto_helper",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/stream_executor:stream",
    ],
)

cc_library(
    name = "tpu_computation_placer",
    srcs = ["tpu_computation_placer.cc"],
    hdrs = ["tpu_computation_placer.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        ":tpu_platform",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:computation_placer",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_platform_interface",
    srcs = ["tpu_platform_interface.cc"],
    hdrs = ["tpu_platform_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "tpu_stream_interface",
    hdrs = ["tpu_stream_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "tpu_executor_interface",
    hdrs = ["tpu_executor_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_platform_interface",
        "//tensorflow/core/platform:errors",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_header",
    ],
)
