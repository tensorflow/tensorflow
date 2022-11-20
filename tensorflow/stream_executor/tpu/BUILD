# Description: StreamExecutor Interface for TPUs

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/brain/experimental/dtensor:__subpackages__",
        "//learning/brain/google/xla/kernels:__subpackages__",
        "//learning/brain/tfrc/executor:__subpackages__",
        "//tensorflow/compiler/jit:__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/compiler/xla/backends/profiler/tpu:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
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
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "c_api_conversions",
    hdrs = ["c_api_conversions.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
    ],
)

cc_library(
    name = "noncopyable_buffer",
    hdrs = ["noncopyable_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:noncopyable_buffer",
    ],
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:status_helper",
    ],
)

cc_library(
    name = "proto_helper",
    hdrs = ["proto_helper.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:proto_helper",
    ],
)

cc_library(
    name = "tpu_executor",
    hdrs = [
        "tpu_executor.h",
        "tpu_platform.h",
        "tpu_stream.h",
        "tpu_timer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_platform_hdr",
    hdrs = ["tpu_platform.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_hdr",
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
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_hdrs",
    ],
)

cc_library(
    name = "tpu_platform_id",
    hdrs = ["tpu_platform_id.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_id",
    ],
)

cc_library(
    name = "tpu_executor_base",
    hdrs = [
        "tpu_event.h",
        "tpu_executor.h",
        "tpu_platform.h",
        "tpu_stream.h",
        "tpu_timer.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_base",
    ],
)

cc_library(
    name = "tpu_node_context",
    hdrs = ["tpu_node_context.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_node_context",
    ],
)

cc_library(
    name = "tpu_transfer_manager_interface",
    hdrs = ["tpu_transfer_manager_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_transfer_manager_interface",
    ],
)

cc_library(
    name = "tpu_transfer_manager",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_transfer_manager",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_transfer_manager_base",
    hdrs = ["tpu_transfer_manager.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_transfer_manager_base",
    ],
)

cc_library(
    name = "tpu_op_executable",
    hdrs = ["tpu_op_executable.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_op_executable",
    ],
)

cc_library(
    name = "tpu_platform_interface",
    hdrs = ["tpu_platform_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_interface",
    ],
)

cc_library(
    name = "tpu_executor_interface",
    hdrs = ["tpu_executor_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_interface",
    ],
)

cc_library(
    name = "tpu_stream_interface",
    hdrs = ["tpu_stream_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_stream_interface",
    ],
)

cc_library(
    name = "tpu_executable_interface",
    hdrs = ["tpu_executable_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executable_interface",
    ],
)

cc_library(
    name = "tpu_executable",
    hdrs = ["tpu_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executable",
    ],
)

cc_library(
    name = "tpu_topology_external",
    hdrs = ["tpu_topology.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_topology_external",
    ],
)
