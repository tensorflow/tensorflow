# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

package(
    default_visibility = [":friends"],
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "launch_dim",
    hdrs = [
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":launch_dim",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "event",
    srcs = [
        "blas.h",
        "device_description.h",
        "device_options.h",
        "dnn.h",
        "event.cc",
        "fft.h",
        "kernel_cache_config.h",
        "launch_dim.h",
        "plugin.h",
        "plugin_registry.h",
        "rng.h",
        "shared_memory_config.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "trace_listener.h",
    ],
    hdrs = [
        "device_memory.h",
        "event.h",
        "kernel.h",
        "kernel_spec.h",
        "platform.h",
        "stream.h",
        "stream_executor_internal.h",
    ],
    deps = [
        ":allocator_stats",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "kernel",
    srcs = [
        "dnn.h",
        "fft.h",
        "kernel.cc",
        "plugin.h",
        "rng.h",
        "stream.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
    ],
    hdrs = [
        "blas.h",
        "device_description.h",
        "device_options.h",
        "event.h",
        "kernel.h",
        "kernel_spec.h",
        "launch_dim.h",
        "multi_platform_manager.h",
        "platform.h",
        "plugin_registry.h",
        "shared_memory_config.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [
        ":allocator_stats",
        ":device_memory",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        ":kernel_cache_config",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_cache_config",
    hdrs = ["kernel_cache_config.h"],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "shared_memory_config",
    hdrs = ["shared_memory_config.h"],
)

cc_library(
    name = "stream_header",
    hdrs = [
        "blas.h",
        "device_memory.h",
        "dnn.h",
        "event.h",
        "fft.h",
        "gpu_launch_dim.h",
        "kernel.h",
        "kernel_cache_config.h",
        "launch_dim.h",
        "stream.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator_stats",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# It implements :stream_header
cc_library(
    name = "stream",
    srcs = [
        "stream.cc",
    ],
    hdrs = ["stream.h"],
    deps = [
        ":blas",
        ":device_memory",
        ":dnn",
        ":event",
        ":fft",
        ":host_or_device_scalar",
        ":kernel",
        ":launch_dim",
        ":platform",
        ":rng",
        ":stream_executor_headers",
        ":stream_executor_internal",
        ":stream_executor_pimpl",
        ":temporary_memory_manager",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "timer",
    srcs = [
        "device_description.h",
        "kernel_cache_config.h",
        "timer.cc",
    ],
    hdrs = [
        "blas.h",
        "kernel.h",
        "stream.h",
        "stream_executor.h",
        "timer.h",
    ],
    deps = [
        ":host_or_device_scalar",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        ":stream_executor_pimpl_header",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":plugin",
        ":stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "rng",
    srcs = ["rng.cc"],
    hdrs = ["rng.h"],
    deps = ["//tensorflow/stream_executor/platform"],
)

cc_library(
    name = "temporary_device_memory",
    srcs = [
        "event.h",
        "temporary_device_memory.cc",
        "temporary_memory_manager.h",
    ],
    hdrs = ["temporary_device_memory.h"],
    deps = [
        ":device_memory",
        ":stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "temporary_memory_manager",
    srcs = ["temporary_memory_manager.cc"],
    hdrs = ["temporary_memory_manager.h"],
    deps = [
        ":device_memory",
        ":stream_executor_pimpl_header",
        ":stream_header",
        ":temporary_device_memory",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "blas",
    srcs = ["blas.cc"],
    hdrs = ["blas.h"],
    deps = [
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = ["//tensorflow/stream_executor/platform"],
)

cc_library(
    name = "host_or_device_scalar",
    hdrs = ["host_or_device_scalar.h"],
    deps = [
        ":device_memory",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "executor_cache",
    srcs = [
        "device_description.h",
        "device_memory.h",
        "device_options.h",
        "event.h",
        "executor_cache.cc",
        "launch_dim.h",
        "plugin.h",
        "plugin_registry.h",
        "rng.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
    ],
    hdrs = [
        "blas.h",
        "dnn.h",
        "executor_cache.h",
        "fft.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "platform.h",
        "shared_memory_config.h",
        "stream.h",
        "stream_executor_internal.h",
        "trace_listener.h",
    ],
    deps = [
        ":allocator_stats",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "multi_platform_manager",
    srcs = ["multi_platform_manager.cc"],
    hdrs = ["multi_platform_manager.h"],
    deps = [
        ":platform",
        ":stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "plugin",
    srcs = ["plugin.cc"],
    hdrs = ["plugin.h"],
)

cc_library(
    name = "plugin_registry",
    srcs = ["plugin_registry.cc"],
    hdrs = ["plugin_registry.h"],
    deps = [
        ":blas",
        ":dnn",
        ":fft",
        ":multi_platform_manager",
        ":platform",
        ":plugin",
        ":stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "scratch_allocator",
    srcs = ["scratch_allocator.cc"],
    hdrs = ["scratch_allocator.h"],
    deps = [
        ":device_memory",
        ":stream_header",
        ":temporary_device_memory",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    cc_api_version = 2,
    default_header = True,
    provide_cc_alias = True,
)

cc_library(
    name = "dnn",
    srcs = ["dnn.cc"],
    hdrs = ["dnn.h"],
    deps = [
        ":device_memory",
        ":dnn_proto_cc",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "stream_executor_internal",
    srcs = [
        "dnn.h",
        "stream_executor_internal.cc",
    ],
    hdrs = [
        "shared_memory_config.h",
        "stream_executor_internal.h",
    ],
    deps = [
        ":allocator_stats",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":dnn_proto_cc",
        ":kernel",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":plugin_registry",
        ":stream_executor_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "stream_executor_pimpl_header",
    hdrs = [
        "device_description.h",
        "dnn.h",
        "kernel.h",
        "kernel_cache_config.h",
        "shared_memory_config.h",
        "stream_executor_pimpl.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator_stats",
        ":dnn_proto_cc",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# It implements :stream_executor_pimpl_header
cc_library(
    name = "stream_executor_pimpl",
    srcs = ["stream_executor_pimpl.cc"],
    hdrs = ["stream_executor_pimpl.h"],
    deps = [
        ":blas",
        ":executor_cache",
        ":fft",
        ":kernel",
        ":platform",
        ":rng",
        ":stream_executor_headers",
        ":stream_header",
        ":timer",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

# The stream_executor_headers target does not prescribe an implementation.
#
# TODO(b/25131218) this is OBSOLETE/DEPRECATED -- get rid of this target altogether
cc_library(
    name = "stream_executor_headers",
    hdrs = [
        "blas.h",
        "device_description.h",
        "device_memory.h",
        "device_options.h",
        "dnn.h",
        "event.h",
        "executor_cache.h",
        "fft.h",
        "gpu_launch_dim.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "launch_dim.h",
        "module_spec.h",
        "multi_platform_manager.h",
        "platform.h",
        "plugin.h",
        "plugin_registry.h",
        "rng.h",
        "shared_memory_config.h",
        "stream.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "timer.h",
        "trace_listener.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator_stats",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "stream_executor",
    hdrs = ["stream_executor.h"],
    deps = [":stream_executor_headers"] + if_static([":stream_executor_impl"]),
)

cc_library(
    name = "stream_executor_impl",
    deps = [
        ":device_description",
        ":device_memory",
        ":dnn_proto_cc",
        ":dnn_proto_cc_impl",
        ":event",
        ":kernel",
        ":launch_dim",
        ":multi_platform_manager",
        ":platform",
        ":stream",
        ":stream_executor_headers",
        ":stream_executor_pimpl",
        ":timer",
    ],
)

cc_library(
    name = "allocator_stats",
    srcs = [
        "allocator_stats.cc",
    ],
    hdrs = ["allocator_stats.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "stream_test",
    size = "small",
    srcs = ["stream_test.cc"],
    deps = [
        ":stream_executor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/host:host_platform",
    ],
)

alias(
    name = "cuda_platform",
    actual = "//tensorflow/stream_executor/cuda:all_runtime",
)

alias(
    name = "rocm_platform",
    actual = "//tensorflow/stream_executor/rocm:all_runtime",
)
