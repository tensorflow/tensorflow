# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

load("//tensorflow:tensorflow.bzl", "tf_cuda_library", "transitive_hdrs")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "launch_dim",
    hdrs = [
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:launch_dim",
    ],
)

cc_library(
    name = "device_description",
    hdrs = ["device_description.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_description",
    ],
)

cc_library(
    name = "event",
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
        "//tensorflow/compiler/xla/stream_executor:event",
    ],
)

cc_library(
    name = "kernel",
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
        "stream_executor.h",
        "stream_executor_internal.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "kernel_spec",
    hdrs = ["kernel_spec.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:kernel_spec",
    ],
)

cc_library(
    name = "kernel_cache_config",
    hdrs = ["kernel_cache_config.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:kernel_cache_config",
    ],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:module_spec",
    ],
)

# Aliases for backwards compatibility.
alias(
    name = "stream_header",
    actual = ":stream_executor_headers",
)

alias(
    name = "stream",
    actual = ":stream_executor",
)

cc_library(
    name = "timer",
    hdrs = [
        "blas.h",
        "kernel.h",
        "stream.h",
        "stream_executor.h",
        "timer.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:timer",
    ],
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:platform",
    ],
)

cc_library(
    name = "rng",
    hdrs = ["rng.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:rng",
    ],
)

cc_library(
    name = "temporary_device_memory",
    hdrs = ["temporary_device_memory.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:temporary_device_memory",
    ],
)

cc_library(
    name = "temporary_memory_manager",
    hdrs = ["temporary_memory_manager.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:temporary_memory_manager",
    ],
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:fft",
    ],
)

cc_library(
    name = "blas",
    hdrs = ["blas.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:blas",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_memory",
    ],
)

cc_library(
    name = "host_or_device_scalar",
    hdrs = ["host_or_device_scalar.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
    ],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_options",
    ],
)

cc_library(
    name = "executor_cache",
    hdrs = [
        "blas.h",
        "executor_cache.h",
        "fft.h",
        "kernel.h",
        "kernel_cache_config.h",
        "kernel_spec.h",
        "platform.h",
        "stream.h",
        "stream_executor_internal.h",
        "trace_listener.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
    ],
)

cc_library(
    name = "multi_platform_manager",
    hdrs = ["multi_platform_manager.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
    ],
)

cc_library(
    name = "plugin",
    hdrs = ["plugin.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:plugin",
    ],
)

cc_library(
    name = "plugin_registry",
    hdrs = ["plugin_registry.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
    ],
)

cc_library(
    name = "scratch_allocator",
    hdrs = ["scratch_allocator.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
    ],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:data_type",
    ],
)

cc_library(
    name = "dnn",
    hdrs = ["dnn.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:dnn",
    ],
)

cc_library(
    name = "stream_executor_internal",
    hdrs = [
        "stream_executor_internal.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "stream_executor_pimpl_header",
    hdrs = [
        "device_description.h",
        "kernel.h",
        "kernel_cache_config.h",
        "stream_executor_pimpl.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
    ],
)

# It implements :stream_executor_pimpl_header
tf_cuda_library(
    name = "stream_executor_pimpl",
    hdrs = ["stream_executor_pimpl.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl",
    ],
)

# The stream_executor_headers target does not prescribe an implementation.
cc_library(
    name = "stream_executor_headers",
    textual_hdrs = [
        "blas.h",
        "device_description.h",
        "device_memory.h",
        "device_memory_allocator.h",
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
        ":host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "stream_executor",
    textual_hdrs = [
        "blas.h",
        "device_description.h",
        "device_memory.h",
        "device_memory_allocator.h",
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
        "stream.h",
        "stream_executor.h",
        "stream_executor_internal.h",
        "stream_executor_pimpl.h",
        "temporary_device_memory.h",
        "temporary_memory_manager.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [":stream_executor_headers"] + if_static([":stream_executor_impl"]),
)

transitive_hdrs(
    name = "stream_executor_install_hdrs",
    deps = [":stream_executor_headers"],
)

cc_library(
    name = "stream_executor_impl",
    deps = ["//tensorflow/compiler/xla/stream_executor:stream_executor_impl"],
)

cc_library(
    name = "allocator_stats",
    hdrs = ["allocator_stats.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:allocator_stats",
    ],
)

cc_library(
    name = "device_memory_allocator",
    hdrs = ["device_memory_allocator.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
    ],
)

cc_library(
    name = "tf_allocator_adapter",
    hdrs = ["tf_allocator_adapter.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:tf_allocator_adapter",
    ],
)

alias(
    name = "cuda_platform",
    actual = "//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
)

alias(
    name = "rocm_platform",
    actual = "//tensorflow/stream_executor/rocm:all_runtime",
)
