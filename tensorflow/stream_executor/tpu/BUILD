# Description: StreamExecutor Interface for TPUs

package(
    default_visibility = ["//tensorflow/core/tpu:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:tf_attrtype",
        "//tensorflow/c:tf_status",
        "//tensorflow/core/tpu:libtftpu_header",
        "//tensorflow/core/tpu/kernels:tpu_util_c_api_hdrs",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_node_context_c_api_hdrs",
    hdrs = ["tpu_node_context_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tpu:libtftpu_header",
        "//tensorflow/core/tpu/kernels:tpu_util_c_api_hdrs",
    ],
    alwayslink = True,
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu/kernels:tpu_util_c_api_hdrs",
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
        "//tensorflow/stream_executor:device_memory",
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
    name = "device_memory_base_helper",
    hdrs = ["device_memory_base_helper.h"],
    deps = [
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/stream_executor:device_memory",
    ],
)

cc_library(
    name = "tpu_executor_base",
    srcs = [
        "tpu_executor.cc",
        "tpu_platform.cc",
    ],
    hdrs = [
        "tpu_executor.h",
        "tpu_platform.h",
        "tpu_stream.h",
        "tpu_timer.h",
    ],
    deps = [
        ":device_memory_base_helper",
        ":status_helper",
        ":tpu_executor_c_api_hdrs",
        ":tpu_executor_interface",
        ":tpu_platform_interface",
        ":tpu_stream_interface",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_executor",
    srcs = ["tpu_platform_registration.cc"],
    deps = [
        ":tpu_executor_base",
        "//tensorflow/stream_executor/platform",
    ],
    alwayslink = True,
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
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tpu_transfer_manager",
    srcs = ["tpu_transfer_manager_registration.cc"],
    deps = [
        ":tpu_executor",
        ":tpu_executor_base",
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
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor",
    ],
)

cc_library(
    name = "tpu_computation_placer",
    srcs = ["tpu_computation_placer.cc"],
    hdrs = ["tpu_computation_placer.h"],
    deps = [
        ":tpu_executor",
        ":tpu_executor_base",
        ":tpu_executor_c_api_hdrs",
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
    name = "tpu_executor_interface",
    hdrs = ["tpu_executor_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_platform_interface",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "tpu_stream_interface",
    hdrs = ["tpu_stream_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/stream_executor:stream_executor_headers",
    ],
)
