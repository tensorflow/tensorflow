load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "pluggable_device_runtime_headers",
    srcs = [
        "pluggable_device.h",
        "pluggable_device_bfc_allocator.h",
        "pluggable_device_context.h",
        "pluggable_device_factory.h",
        "pluggable_device_init.h",
        "pluggable_device_process_state.h",
        "pluggable_device_util.h",
        "//tensorflow/core/common_runtime/device:device_runtime_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pluggable_device_runtime_impl",
    srcs = [
        "pluggable_device.cc",
        "pluggable_device_context.cc",
        "pluggable_device_factory.cc",
        "pluggable_device_process_state.cc",
        "pluggable_device_util.cc",
    ],
    hdrs = [":pluggable_device_runtime_headers"],
    copts = tf_copts(),
    deps = [
        ":pluggable_device_bfc_allocator",
        ":pluggable_device_init_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/common_runtime/device:device_event_mgr",
        "//tensorflow/core/platform:tensor_float_32_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pluggable_device_plugin_init",
    srcs = [
        "pluggable_device_plugin_init.cc",
    ],
    hdrs = [
        "pluggable_device_plugin_init.h",
        ":pluggable_device_runtime_headers",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
    ],
)

cc_library(
    name = "pluggable_device_runtime",
    hdrs = [":pluggable_device_runtime_headers"],
    linkstatic = 1,
    deps = [
        ":pluggable_device_runtime_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
    ],
)

cc_library(
    name = "pluggable_device_bfc_allocator",
    srcs = [
        "pluggable_device_bfc_allocator.cc",
    ],
    hdrs = ["pluggable_device_bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/device:device_id",
        "//tensorflow/core/common_runtime/device:device_mem_allocator",
    ],
)

cc_library(
    name = "pluggable_device_init",
    hdrs = [
        "pluggable_device_init.h",
    ],
    deps = [
        "pluggable_device_init_impl",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
    ],
)

cc_library(
    name = "pluggable_device_init_impl",
    srcs = [
        "pluggable_device_init.cc",
    ],
    hdrs = [
        "pluggable_device_init.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
    ],
    alwayslink = 1,
)
