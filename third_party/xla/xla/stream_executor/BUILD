load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:build_config_root.bzl", "if_static")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends", "stream_executor_internal")
load("//xla/tsl:tsl.bzl", "internal_visibility", "transitive_hdrs")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

# StreamExecutor clients that can depend on targets defined in stream_executor package.
package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

# StreamExecutor platform-dependent implementations. We restrict visibility of all internal
# implementation interfaces to internal users (everything in `stream_executor::internal` namespace).
package_group(
    name = "internal",
    packages = stream_executor_internal(),
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor public API
#===--------------------------------------------------------------------------------------------===#

# If you want to use StreamExecutor you have to depend on the "core" `xla/stream_executor` target
# defined below which exports StreamExecutor public API headers and also provides implemenetation.

# StreamExecutor also have a small number of public libraries that do not depend on StreamExecutor
# itself (e.g. `device_description`), this is mostly a result of StreamExecutor being a hardware
# abstraction layer for XLA (and Tensorflow).

# StreamExecutor itself is a small abstrtaction layer on top of platform-specific API
# implementations (e.g. see `stream_executor/cuda` folder for CUDA-specific details), and should
# not contribute a lot to binary size or compilation time.

# We bundle headers into filegroups for internal use only (we re-export the same set of headers
# from multiple targets), and all external clients should depend on one of the public `cc_library`
# targets that have dependencies required for compiling headers (e.g. absl dependencies). These
# filegroup roughly correspond to "StreamExecutor components" that are available to the clients.

# These are the headers that constitute StreamExecutor public API. Clients should not depend on
# this filegroup directly, but instead depend on a `stream_executor` target that implements these
# headers. We also have a header-only `stream_executor_headers` target, but currently this is
# an implementation detail of StreamExecutor and has internal visibility.
#
# TODO(ezhulenev): Remove from public API headers that are exported via standalone public libraries,
# e.g. `platform` and `platform_manager` should be added with an explicit dependency.
filegroup(
    name = "stream_executor_api_headers",
    srcs = [
        "allocator_stats.h",
        "command_buffer.h",
        "data_type.h",
        "device_description.h",
        "device_memory.h",
        "device_memory_allocator.h",
        "event.h",
        "executor_cache.h",
        "host_memory_allocation.h",
        "kernel.h",
        "kernel_spec.h",
        "launch_dim.h",
        "memory_allocation.h",
        "module_spec.h",
        "numeric_options.h",
        "platform.h",
        "platform_manager.h",
        "scratch_allocator.h",
        "stream.h",
        "stream_executor.h",
    ],
    visibility = ["//visibility:private"],
)

# These are the headers for default StreamExecutor plugins.
filegroup(
    name = "stream_executor_plugin_headers",
    srcs = [
        "blas.h",
        "dnn.h",
        "fft.h",
    ],
    visibility = ["//visibility:private"],
)

# This is a list of dependencies required for building `stream_executor` target (and required for
# making `stream_executor_headers` self-contained, which means that you can include any of the
# public API headers and don't worry about adding dependencies).
STREAM_EXECUTOR_DEPENDENCIES = [
    ":device_description_proto_cc",
    ":host_or_device_scalar",
    "@com_google_absl//absl/algorithm:container",
    "@com_google_absl//absl/base:core_headers",
    "@com_google_absl//absl/container:inlined_vector",
    "@com_google_absl//absl/meta:type_traits",
    "@com_google_absl//absl/container:node_hash_map",
    "@com_google_absl//absl/functional:any_invocable",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/synchronization",
    "@com_google_absl//absl/types:span",
    "//xla/stream_executor/platform",
    "@local_tsl//tsl/framework:device_id",
    "@local_tsl//tsl/framework:device_type",
    "@local_tsl//tsl/lib/gtl:int_type",
    "@local_tsl//tsl/platform:env",
    "@local_tsl//tsl/platform:errors",
    "@local_tsl//tsl/platform:ml_dtypes",
    "@local_tsl//tsl/platform:logging",
    "@local_tsl//tsl/platform:status",
    "@local_tsl//tsl/platform:statusor",
    "@local_tsl//tsl/platform:types",
    "@local_tsl//tsl/protobuf:dnn_proto_cc",
]

cc_library(
    name = "stream_executor",
    hdrs = [
        ":stream_executor_api_headers",
        ":stream_executor_plugin_headers",
    ],
    deps = STREAM_EXECUTOR_DEPENDENCIES + [
        ":stream_executor_pimpl",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
    ] + if_static([
        ":stream_executor_impl",
        "@com_google_protobuf//:protobuf",  # indirectly-used by dnn.h
    ]),
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor public libraries
#===--------------------------------------------------------------------------------------------===#

# Some of the StreamExecutor libraries that do not depend on StreamExecutor itself (Stream, Kernel,
# Event, etc.) exported as standalone libraries (these libraries should not depend on
# `stream_executor` and `stream_executor_headers` targets). This is mostly a historical artifact of
# an era when StreamExecutor was a part of Tensorflow.

tf_proto_library(
    name = "device_description_proto",
    srcs = ["device_description.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = ["//xla:autotune_results_proto"],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":device_description_proto_cc",
        ":launch_dim",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = ["@local_tsl//tsl/platform:logging"],
)

cc_library(
    name = "device_memory_handle",
    srcs = ["device_memory_handle.cc"],
    hdrs = ["device_memory_handle.h"],
    deps = [
        ":device_memory",
        ":stream_executor_interface",
    ],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "scoped_module_handle",
    hdrs = ["scoped_module_handle.h"],
    deps = [
        ":module_spec",
        ":stream_executor_interface",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "mock_stream_executor",
    testonly = True,
    hdrs = ["mock_stream_executor.h"],
    deps = [
        ":device_description",
        ":device_memory",
        ":event_interface",
        ":kernel_spec",
        ":module_spec",
        ":stream_executor_headers",
        ":stream_executor_interface",
        ":stream_interface",
        "//xla:test",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ] + if_static(["@com_google_protobuf//:protobuf"]),
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    visibility = [":internal"],
    deps = [
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ],
)

cc_library(
    name = "device_memory_allocator",
    srcs = ["device_memory_allocator.cc"],
    hdrs = ["device_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":platform",
        ":stream_executor_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_memory_allocation",
    srcs = ["host_memory_allocation.cc"],
    hdrs = ["host_memory_allocation.h"],
    deps = [
        ":memory_allocation",
        ":stream_executor_interface",
    ],
)

cc_library(
    name = "host_or_device_scalar",
    hdrs = ["host_or_device_scalar.h"],
    deps = [
        ":device_memory",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "launch_dim",
    hdrs = ["launch_dim.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "memory_allocation",
    hdrs = ["memory_allocation.h"],
)

cc_library(
    name = "platform_manager",
    srcs = ["platform_manager.cc"],
    hdrs = ["platform_manager.h"],
    deps = [
        ":platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "numeric_options",
    hdrs = ["numeric_options.h"],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":device_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor plugins
#===--------------------------------------------------------------------------------------------===#

# TODO(ezhulenev): Today all StreamExecutor plugins are linked by default into the `stream_executor`
# target and leak into "core" APIs. We should decouple all plugins into optional dependencies, and
# make sure that they are not exposed via "core" APIs (se::Stream, se::StreamExecutor, etc.).

cc_library(
    name = "blas",
    srcs = ["blas.cc"],
    hdrs = ["blas.h"],
    deps = [
        ":data_type",
        ":device_memory",
        ":numeric_options",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
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
        ":numeric_options",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",  # buildcleaner: keep
        "@local_tsl//tsl/lib/strings:proto_serialization",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ] + if_static(["@com_google_protobuf//:protobuf"]),
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
    deps = [
        "//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "lazy_op_runner",
    hdrs = ["lazy_op_runner.h"],
    deps = [
        ":stream_executor_headers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ],
)

# TODO(ezhulenev): This should be removed.
exports_files(["lazy_op_runner.h"])

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor platform-dependent interfaces
#===--------------------------------------------------------------------------------------------===#

# Only platform-dependent StreamExecutor implementations (e.g. StreamExecutor for GPUs) and targets
# defined by StreamExecutor itself (e.g. `event`, `kernel`, etc.) can depend on internal
# interfaces (interfaces that define platform-specific API).
#
# External clients of StreamExecutor should depend on `stream_executor` target (links StreamExecutor
# implementation in static build configuration), or a header only `stream_executor_headers`.

cc_library(
    name = "stream_executor_interface",
    hdrs = [
        "stream_executor_interface.h",
    ],
    deps = [
        ":stream_executor_headers",
        ":stream_interface",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor internal implementation (visible to StreamExecutor platform implementations)
#===--------------------------------------------------------------------------------------------===#

# This header-only library is an internal implementation detail of StreamExecutor (to break
# dependency cycles between internal libraries). All StreamExecutor clients should depend on a
# regular `stream_executor` target that also provides an implementation for all headers.
cc_library(
    name = "stream_executor_headers",
    hdrs = [
        "stream_executor_interface.h",
        "stream_executor_pimpl.h",  # TODO(301020144): Remove internal header
        ":stream_executor_api_headers",
        ":stream_executor_plugin_headers",
    ],
    visibility = [":internal"],
    deps = STREAM_EXECUTOR_DEPENDENCIES + if_static([
        "@com_google_protobuf//:protobuf",  # indirectly-used by dnn.h
    ]) + [
        ":stream_interface",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "plugin_registry",
    srcs = ["plugin_registry.cc"],
    hdrs = ["plugin_registry.h"],
    visibility = [":internal"],
    deps = [
        ":blas",
        ":dnn",
        ":fft",
        ":platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "event_interface",
    hdrs = ["event_interface.h"],
    deps = [
    ],
)

cc_library(
    name = "stream_interface",
    hdrs = ["stream_interface.h"],
    deps = [
        ":platform",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor private implementation (has private visibility)
#===--------------------------------------------------------------------------------------------===#

# Targets that implement StreamExecutor APIs are private, and should not be used outside of
# `stream_executor` package. Clients should depend on `stream_executor` (headers and
# implementation) or `stream_executor_headers` (only headers, if there is a reason not to link
# implementation) if they want to use StreamExecutor.

cc_library(
    name = "allocator_stats",
    srcs = ["allocator_stats.cc"],
    hdrs = ["allocator_stats.h"],
    visibility = ["//visibility:private"],
    deps = ["@com_google_absl//absl/strings:str_format"],
)

cc_library(
    name = "command_buffer",
    srcs = ["command_buffer.cc"],
    hdrs = ["command_buffer.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":stream_executor_headers",
        ":stream_executor_interface",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/gtl:int_type",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "event",
    srcs = ["event.cc"],
    hdrs = ["event.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":event_interface",
        ":stream_executor_headers",
        ":stream_executor_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "executor_cache",
    srcs = ["executor_cache.cc"],
    hdrs = ["executor_cache.h"],
    deps = [
        ":platform",
        ":stream_executor_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [
        ":device_memory",
        ":kernel_spec",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_interface",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "scratch_allocator",
    hdrs = ["scratch_allocator.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":device_memory_allocator",
        ":stream_executor_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

#===--------------------------------------------------------------------------------------------===#

transitive_hdrs(
    name = "stream_executor_install_hdrs",
    tags = [
        "alt_dep=:stream_executor_headers",
        "avoid_dep",
    ],
    deps = [":stream_executor_headers"],
)

# TODO(ezhulenev): This should be merged into regular `stream_executor` target and `stream.cc` moved
# into its own target, however today we have problems with backward references when we try to link
# everything together. See: https://lld.llvm.org/ELF/warn_backrefs.html.
cc_library(
    name = "stream_executor_pimpl",
    srcs = [
        "stream.cc",
        "stream_executor_pimpl.cc",
    ],
    hdrs = ["stream_executor_pimpl.h"],
    tags = ["avoid_dep"],
    visibility = ["//visibility:private"],
    deps = [
        ":blas",  # build_cleaner: keep
        ":command_buffer",  # build_cleaner: keep
        ":dnn",  # build_cleaner: keep
        ":fft",
        ":host_memory_allocation",  # build_cleaner: keep
        ":kernel_spec",
        ":platform",
        ":stream_executor_headers",
        ":stream_executor_interface",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# We have a separate `stream_executor_impl` target because in open source we are building multiple
# shared libraries and then link them together (this is an implementation detail of Tensorflow
# framework), and we take extra care not to define symbols in multiple objects. Otherwise we can
# end up with static singletons declared in multiple objects, ODR violations, and many other bad
# things that lead to nearly impossible to debug run time crashes.
cc_library(
    name = "stream_executor_impl",
    deps = [
        ":allocator_stats",
        ":device_description",
        ":device_memory",
        ":dnn",
        ":event",
        ":executor_cache",
        ":kernel",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":platform_manager",
        ":scratch_allocator",
        ":stream_executor_headers",
        ":stream_executor_pimpl",
        "@local_tsl//tsl/protobuf:dnn_proto_cc_impl",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor tests
#===--------------------------------------------------------------------------------------------===#

xla_cc_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    deps = [
        ":device_memory",
        ":stream_executor",
        "//xla/stream_executor/host:host_platform",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "stream_executor_test",
    srcs = ["stream_executor_test.cc"],
    deps = [
        ":stream_executor",
        "//xla/stream_executor/host:host_platform",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "stream_test",
    size = "small",
    srcs = ["stream_test.cc"],
    deps = [
        ":stream_executor",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "dnn_test",
    size = "small",
    srcs = ["dnn_test.cc"],
    deps = [
        ":dnn",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "scoped_module_handle_test",
    srcs = ["scoped_module_handle_test.cc"],
    deps = [
        ":device_description",
        ":mock_stream_executor",
        ":module_spec",
        ":scoped_module_handle",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "device_memory_handle_test",
    srcs = ["device_memory_handle_test.cc"],
    deps = [
        ":device_memory",
        ":device_memory_handle",
        ":mock_stream_executor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# Aliases for StreamExecutor platforms
#===--------------------------------------------------------------------------------------------===#

alias(
    name = "cuda_platform",
    actual = "//xla/stream_executor/cuda:all_runtime",
)

alias(
    name = "rocm_platform",
    actual = "//xla/stream_executor/rocm:all_runtime",
)

# TODO(ezhulenev): This should be removed.
cc_library(
    name = "stream_executor_bundle",
    visibility = [":internal"],
    deps = [
        ":dnn",
        ":event",
        ":platform_manager",
        ":scratch_allocator",
        ":stream_executor",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/platform:dso_loader",
        "//xla/stream_executor/rocm:rocm_platform_id",
    ],
)
