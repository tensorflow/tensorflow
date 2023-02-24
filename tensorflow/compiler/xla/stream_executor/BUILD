# GPU executor library for data-parallel kernel launches and cross-platform
# HPC-library APIs.
#
# Throughout this file, all targets are built with the standard crosstool and
# do not link against restricted binary blobs.

load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load("//tensorflow/tsl:tsl.bzl", "transitive_hdrs", "tsl_gpu_library")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
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
        ":device_description_proto_cc",
        ":dnn_proto_cc",
        ":host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

transitive_hdrs(
    name = "stream_executor_install_hdrs",
    deps = [":stream_executor_headers"],
)

cc_library(
    name = "launch_dim",
    hdrs = [
        "gpu_launch_dim.h",
        "launch_dim.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_listener",
    hdrs = [
        "trace_listener.h",
    ],
    deps = [
        ":device_memory",
        ":kernel",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "device_description_proto",
    srcs = ["device_description.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/compiler/xla:autotune_results_proto",
    ],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":device_description_proto_cc",
        ":launch_dim",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/lib/math:math_util",
        "//tensorflow/tsl/platform:numbers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
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
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "allocator_stats",
    srcs = [
        "allocator_stats.cc",
    ],
    hdrs = ["allocator_stats.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":dnn_proto_cc",
        "//tensorflow/tsl/platform:float8",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
    ],
)

cc_library(
    name = "device_id_utils",
    hdrs = ["device_id_utils.h"],
    deps = [
        ":platform",
        ":stream_executor",
        "//tensorflow/tsl/framework:device_id_impl",
        "//tensorflow/tsl/lib/gtl:int_type",
        "//tensorflow/tsl/platform:str_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/platform"],
)

cc_library(
    name = "device_options",
    hdrs = ["device_options.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    make_default_target_header_only = True,
    protodeps = ["//tensorflow/tsl/protobuf:dnn_proto"],
    exports = ["//tensorflow/tsl/protobuf:dnn_proto"],
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":plugin",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "plugin",
    srcs = ["plugin.cc"],
    hdrs = ["plugin.h"],
)

cc_library(
    name = "rng",
    srcs = ["rng.cc"],
    hdrs = ["rng.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "device_mem_allocator",
    hdrs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/framework:device_id",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/profiler/lib:traceme",
    ],
)

filegroup(
    name = "device_mem_allocator_headers",
    srcs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
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
        ":data_type",
        ":device_description",
        ":device_description_proto_cc",
        ":device_memory",
        ":device_options",
        ":fft",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":plugin",
        ":rng",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
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
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "stream_executor_internal",
    hdrs = [
        "stream_executor_internal.h",
    ],
    deps = [
        ":allocator_stats",
        ":device_description",
        ":device_memory",
        ":device_options",
        ":kernel",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":plugin_registry",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/types:optional",
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
        ":device_description",
        ":device_description_proto_cc",
        ":kernel_cache_config",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# It implements :stream_executor_pimpl_header
tsl_gpu_library(
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
        ":plugin_registry",
        ":rng",
        ":stream_executor_headers",
        ":stream_executor_internal",
        ":temporary_device_memory",
        ":temporary_memory_manager",
        ":timer",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:stacktrace",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "//tensorflow/tsl/util:env_var",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
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
        ":data_type",
        ":device_description",
        ":device_description_proto_cc",
        ":device_memory",
        ":device_options",
        ":fft",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":plugin",
        ":plugin_registry",
        ":rng",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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
        ":data_type",
        ":device_description",
        ":device_description_proto_cc",
        ":kernel_cache_config",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "blas",
    srcs = ["blas.cc"],
    hdrs = ["blas.h"],
    deps = [
        ":data_type",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_or_device_scalar",
    hdrs = ["host_or_device_scalar.h"],
    deps = [
        ":data_type",
        ":device_memory",
        "//tensorflow/compiler/xla/stream_executor/platform",
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
        ":blas",
        ":data_type",
        ":device_description",
        ":device_description_proto_cc",
        ":device_memory",
        ":device_options",
        ":fft",
        ":kernel_cache_config",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":plugin",
        ":rng",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
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
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "dnn",
    srcs = ["dnn.cc"],
    hdrs = ["dnn.h"],
    deps = [
        ":data_type",
        ":device_description_proto_cc",
        ":device_memory",
        ":dnn_proto_cc",
        ":stream_executor_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/lib/strings:proto_serialization",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
    ] + if_static(["@com_google_protobuf//:protobuf"]),
)

cc_library(
    name = "temporary_memory_manager",
    srcs = ["temporary_memory_manager.cc"],
    hdrs = ["temporary_memory_manager.h"],
    deps = [
        ":device_memory",
        ":stream_executor_headers",
        ":stream_executor_pimpl_header",
        ":temporary_device_memory",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
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
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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
    deps = [
        ":stream_executor_headers",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
    ] + if_static([":stream_executor_impl"]),
)

cc_library(
    name = "stream_executor_impl",
    deps = [
        ":device_description",
        ":device_memory",
        ":dnn_proto_cc",
        ":event",
        ":kernel",
        ":launch_dim",
        ":multi_platform_manager",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_pimpl",
        ":timer",
        "//tensorflow/compiler/xla/stream_executor:dnn_proto_cc_impl",
        "//tensorflow/tsl/protobuf:dnn_proto_cc_impl",
    ],
)

cc_library(
    name = "device_memory_allocator",
    hdrs = ["device_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":platform",
        ":stream_executor",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scratch_allocator",
    srcs = ["scratch_allocator.cc"],
    hdrs = ["scratch_allocator.h"],
    deps = [
        ":device_memory",
        ":stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

xla_cc_test(
    name = "stream_test",
    size = "small",
    srcs = ["stream_test.cc"],
    deps = [
        ":stream_executor",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "dnn_test",
    size = "small",
    srcs = ["dnn_test.cc"],
    deps = [
        ":dnn",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "lazy_op_runner",
    hdrs = ["lazy_op_runner.h"],
    deps = [
        ":dnn",
        ":stream_executor_headers",
        "@com_google_absl//absl/base",
    ],
)

exports_files(["lazy_op_runner.h"])

alias(
    name = "cuda_platform",
    actual = "//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
)

alias(
    name = "rocm_platform",
    actual = "//tensorflow/compiler/xla/stream_executor/rocm:all_runtime",
)

# TODO(se-owner): document or remove this.
cc_library(
    name = "stream_executor_bundle",
    deps = [
        ":dnn",
        ":event",
        ":multi_platform_manager",
        ":scratch_allocator",
        ":stream_executor",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
    ],
)
