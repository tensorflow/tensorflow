# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

load("//tensorflow:tensorflow.bzl", "filegroup", "tf_cc_test", "tf_cuda_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
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
        "event.cc",
        "fft.h",
        "kernel_cache_config.h",
        "launch_dim.h",
        "plugin.h",
        "plugin_registry.h",
        "rng.h",
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
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

MATH_DEPS = [
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_grad",
    "//tensorflow/core/framework:bounds_check",
    "//third_party/eigen3",
]

cc_library(
    name = "matmul_util",
    srcs = ["matmul_util.cc"],
    hdrs = ["matmul_util.h"],
    deps = [
        ":stream_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core/kernels:gpu_utils",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core:protos_all_cc",
    ] + if_static(
        ["//tensorflow/core/platform:tensor_float_32_utils"],
        ["//tensorflow/core/platform:tensor_float_32_hdr_lib"],
    ) + MATH_DEPS,
)

cc_library(
    name = "kernel",
    srcs = [
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
        "stream_executor.h",
        "stream_executor_internal.h",
        "timer.h",
        "trace_listener.h",
    ],
    deps = [
        ":allocator_stats",
        ":device_memory",
        ":host_or_device_scalar",
        ":kernel_cache_config",
        ":stream_executor_headers",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:variant",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
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
        ":data_type",
        ":device_memory",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
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
        ":allocator_stats",
        ":host_or_device_scalar",
        ":stream_executor_headers",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
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
        "@com_google_absl//absl/strings:str_format",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "scratch_allocator",
    srcs = ["scratch_allocator.cc"],
    hdrs = ["scratch_allocator.h"],
    deps = [
        ":device_memory",
        ":stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":dnn_proto_cc",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "dnn",
    srcs = ["dnn.cc"],
    hdrs = ["dnn.h"],
    deps = [
        ":data_type",
        ":device_memory",
        ":dnn_proto_cc",
        ":stream_executor_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ] + if_static(["@com_google_protobuf//:protobuf"]),
)

cc_library(
    name = "lazy_op_runner",
    hdrs = ["lazy_op_runner.h"],
    deps = [
        ":dnn",
        ":stream_header",
    ],
)

# This is terrible, but we need to defeat the dependency check to depend on the
# header-only lazy_op_runner template in fused_conv2d_bias_activation (being
# header-only means it can't cause linkage issues), so export the actual header
# file itself and add a target for it in core/kernels.  Unsophisticated checks
# beget dumb circumventions :(
#
# TODO(b/202546057) get rid of this once fused_conv is moved out of contrib.
exports_files(["lazy_op_runner.h"])

cc_library(
    name = "stream_executor_internal",
    srcs = [
        "stream_executor_internal.cc",
    ],
    hdrs = [
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
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

# It implements :stream_executor_pimpl_header
tf_cuda_library(
    name = "stream_executor_pimpl",
    srcs = [
        "stream.cc",
        "stream_executor_pimpl.cc",
    ],
    hdrs = ["stream_executor_pimpl.h"],
    deps = [
        ":blas",
        ":device_memory",
        ":dnn",
        ":event",
        ":executor_cache",
        ":fft",
        ":host_or_device_scalar",
        ":kernel",
        ":launch_dim",
        ":platform",
        ":rng",
        ":stream_executor_headers",
        ":stream_executor_internal",
        ":stream_header",
        ":temporary_memory_manager",
        ":timer",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/cuda:cuda_dnn_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        ":allocator_stats",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        "//tensorflow/core/platform:logging",
        "//tensorflow/stream_executor/cuda:cuda_dnn_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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

cc_library(
    name = "device_memory_allocator",
    hdrs = ["device_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":platform",
        ":stream_executor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_allocator_adapter",
    srcs = ["tf_allocator_adapter.cc"],
    hdrs = ["tf_allocator_adapter.h"],
    deps = [
        ":device_memory",
        ":device_memory_allocator",
        ":platform",
        ":stream_executor_headers",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:errors",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/synchronization",
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
