load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_build_defs_bzl_deps", "stream_executor_friends", "stream_executor_internal")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "//xla/tsl/platform:rules_cc_bzl",
        "//xla/tsl/platform/default:cuda_build_defs_bzl",
        "@local_config_cuda//cuda:build_defs_bzl",
        "@local_xla//third_party/py/rules_pywrap:pywrap_bzl",
    ] + stream_executor_build_defs_bzl_deps(),
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor public libraries
#===--------------------------------------------------------------------------------------------===#

tf_proto_library(
    name = "device_description_proto",
    srcs = ["device_description.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//xla:autotune_results_proto",
        "//xla/stream_executor/cuda:cuda_compute_capability_proto",
    ],
)

cc_library(
    name = "activate_context",
    hdrs = ["activate_context.h"],
)

cc_library(
    name = "device_description",
    srcs = ["device_description.cc"],
    hdrs = ["device_description.h"],
    deps = [
        ":device_description_proto_cc",
        ":launch_dim",
        ":semantic_version",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/lib/math:math_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "device_memory",
    hdrs = ["device_memory.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "device_memory_handle",
    srcs = ["device_memory_handle.cc"],
    hdrs = ["device_memory_handle.h"],
    deps = [
        ":device_memory",
        ":stream_executor_h",
    ],
)

cc_library(
    name = "module_spec",
    hdrs = ["module_spec.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "scoped_module_handle",
    hdrs = ["scoped_module_handle.h"],
    deps = [
        ":module_spec",
        ":stream_executor_h",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "gpu_solver_context",
    hdrs = ["gpu_solver_context.h"],
    deps = [
        ":blas",
        ":device_memory",
        ":stream",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "stream_finder",
    srcs = ["stream_finder.cc"],
    hdrs = ["stream_finder.h"],
    deps = [
        ":platform",
        ":stream",
        ":stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "mock_platform",
    testonly = True,
    hdrs = ["mock_platform.h"],
    deps = [
        ":device_description",
        ":platform",
        ":stream_executor_h",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "mock_stream",
    testonly = True,
    hdrs = ["mock_stream.h"],
    deps = [
        ":device_description",
        ":device_memory",
        ":event",
        ":event_based_timer",
        ":launch_dim",
        ":platform",
        ":stream",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "mock_stream_executor",
    testonly = True,
    hdrs = ["mock_stream_executor.h"],
    deps = [
        ":activate_context",
        ":allocator_stats",
        ":blas",
        ":command_buffer",
        ":device_description",
        ":device_memory",
        ":dnn",
        ":event",
        ":event_based_timer",
        ":fft",
        ":kernel",
        ":kernel_spec",
        ":launch_dim",
        ":memory_allocation",
        ":memory_allocator",
        ":module_spec",
        ":platform",
        ":stream",
        ":stream_executor_h",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ] + if_google(["@com_google_protobuf//:wrappers_cc_proto"]),
)

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "device_memory_allocator",
    hdrs = ["device_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":platform",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "stream_executor_memory_allocator",
    srcs = ["stream_executor_memory_allocator.cc"],
    hdrs = ["stream_executor_memory_allocator.h"],
    deps = [
        ":device_memory",
        ":device_memory_allocator",
        ":platform",
        ":stream",
        ":stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
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

cc_library(
    name = "blas",
    srcs = ["blas.cc"],
    hdrs = ["blas.h"],
    deps = [
        ":blas_proto_cc",
        ":data_type",
        ":device_memory",
        ":numeric_options",
        ":scratch_allocator",
        ":stream",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
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
        ":scratch_allocator",
        ":stream",
        "//xla:util",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/protobuf:dnn_proto_cc",
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
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_google(["@com_google_protobuf//:wrappers_cc_proto"]),
)

tf_proto_library(
    name = "blas_proto",
    srcs = ["blas.proto"],
)

cc_library(
    name = "fft",
    hdrs = ["fft.h"],
)

cc_library(
    name = "lazy_op_runner",
    hdrs = ["lazy_op_runner.h"],
    deps = [
        ":dnn",
        ":stream",
        ":stream_executor_h",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "stream_executor_h",
    hdrs = [
        "stream_executor.h",
    ],
    deps = [
        ":activate_context",
        ":allocator_stats",
        ":blas",
        ":command_buffer",
        ":device_description",
        ":device_memory",
        ":dnn",
        ":event",
        ":event_based_timer",
        ":fft",
        ":kernel",
        ":kernel_spec",
        ":memory_allocation",
        ":memory_allocator",
        ":module_spec",
        ":platform",
        ":stream",
        "//xla/stream_executor/gpu:tma_metadata",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "event",
    hdrs = ["event.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "memory_allocator",
    hdrs = ["memory_allocator.h"],
    deps = [
        ":memory_allocation",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "generic_memory_allocator",
    hdrs = ["generic_memory_allocator.h"],
    deps = [
        ":memory_allocation",
        ":memory_allocator",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "generic_memory_allocation",
    hdrs = ["generic_memory_allocation.h"],
    deps = [
        ":memory_allocation",
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

cc_library(
    name = "stream",
    hdrs = ["stream.h"],
    deps = [
        ":device_description",
        ":device_memory",
        ":event",
        ":event_based_timer",
        ":launch_dim",
        ":platform",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "event_based_timer",
    hdrs = ["event_based_timer.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "executor_cache",
    srcs = ["executor_cache.cc"],
    hdrs = ["executor_cache.h"],
    deps = [
        ":stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        ":launch_dim",
        ":stream",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "typed_kernel_factory",
    hdrs = ["typed_kernel_factory.h"],
    deps = [
        ":kernel",
        ":kernel_spec",
        ":stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "semantic_version",
    srcs = ["semantic_version.cc"],
    hdrs = ["semantic_version.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "scratch_allocator",
    hdrs = ["scratch_allocator.h"],
    deps = [
        ":device_memory",
        ":device_memory_allocator",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor internal implementation (visible to StreamExecutor platform implementations)
#===--------------------------------------------------------------------------------------------===#

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
    name = "allocator_stats",
    srcs = ["allocator_stats.cc"],
    hdrs = ["allocator_stats.h"],
    deps = ["@com_google_absl//absl/strings:str_format"],
)

cc_library(
    name = "command_buffer",
    hdrs = ["command_buffer.h"],
    visibility = [":internal"],
    deps = [
        ":bit_pattern",
        ":device_memory",
        ":dnn",
        ":kernel",
        ":launch_dim",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "trace_command_buffer_factory",
    srcs = ["trace_command_buffer_factory.cc"],
    hdrs = ["trace_command_buffer_factory.h"],
    visibility = [":internal"],
    deps = [
        ":command_buffer",
        ":stream",
        ":stream_executor_h",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "stream_executor_common",
    srcs = ["stream_executor_common.cc"],
    hdrs = ["stream_executor_common.h"],
    deps = [
        ":activate_context",
        ":device_description",
        ":platform",
        ":stream_executor_h",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "stream_common",
    srcs = ["stream_common.cc"],
    hdrs = ["stream_common.h"],
    deps = [
        ":device_description",
        ":fft",
        ":platform",
        ":stream",
        ":stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "bit_pattern",
    srcs = ["bit_pattern.cc"],
    hdrs = ["bit_pattern.h"],
    visibility = [":internal"],
    deps = ["@com_google_absl//absl/strings"],
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
        ":blas",  # build_cleaner: keep
        ":command_buffer",  # build_cleaner: keep
        ":device_description",
        ":device_memory",
        ":dnn",
        ":executor_cache",
        ":fft",
        ":kernel",
        ":kernel_spec",
        ":launch_dim",
        ":platform",
        ":platform_manager",
        ":scratch_allocator",
        ":stream_common",
        ":stream_executor_common",
        ":stream_executor_h",
    ] + if_oss(["//xla/tsl/protobuf:dnn_proto_cc_impl"]),
)

#===--------------------------------------------------------------------------------------------===#
# StreamExecutor tests
#===--------------------------------------------------------------------------------------------===#

xla_cc_test(
    name = "generic_memory_allocation_test",
    srcs = ["generic_memory_allocation_test.cc"],
    deps = [
        ":generic_memory_allocation",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "generic_memory_allocator_test",
    srcs = ["generic_memory_allocator_test.cc"],
    deps = [
        ":generic_memory_allocation",
        ":generic_memory_allocator",
        ":memory_allocation",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "semantic_version_test",
    srcs = ["semantic_version_test.cc"],
    deps = [
        ":semantic_version",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    deps = [
        ":device_memory",
        ":kernel",
        ":kernel_spec",
        ":platform",
        ":platform_manager",
        ":stream_executor_h",
        ":typed_kernel_factory",
        "//xla/stream_executor/host:host_platform",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

xla_cc_test(
    name = "stream_executor_test",
    srcs = ["stream_executor_test.cc"],
    deps = [
        ":platform",
        ":platform_manager",
        ":stream_executor_h",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "stream_test",
    size = "small",
    srcs = ["stream_test.cc"],
    deps = [
        ":platform",
        ":platform_manager",
        ":stream",
        ":stream_executor_h",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "dnn_test",
    size = "small",
    srcs = ["dnn_test.cc"],
    deps = [
        ":dnn",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "scoped_module_handle_test",
    srcs = ["scoped_module_handle_test.cc"],
    deps = [
        ":mock_stream_executor",
        ":module_spec",
        ":scoped_module_handle",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "device_memory_handle_test",
    srcs = ["device_memory_handle_test.cc"],
    deps = [
        ":device_memory",
        ":device_memory_handle",
        ":mock_stream_executor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "device_description_test",
    srcs = ["device_description_test.cc"],
    deps = [
        ":device_description",
        ":semantic_version",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "executor_cache_test",
    srcs = ["executor_cache_test.cc"],
    deps = [
        ":executor_cache",
        ":mock_stream_executor",
        ":stream",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "stream_finder_test",
    srcs = ["stream_finder_test.cc"],
    deps = [
        ":mock_platform",
        ":mock_stream",
        ":mock_stream_executor",
        ":stream_finder",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "bit_pattern_test",
    srcs = ["bit_pattern_test.cc"],
    deps = [
        ":bit_pattern",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# Aliases for StreamExecutor platforms
#===--------------------------------------------------------------------------------------------===#

alias(
    name = "cuda_platform",
    actual = "//xla/stream_executor/cuda:all_runtime",
    tags = [
        "cuda-only",
        "gpu",
    ],
)

alias(
    name = "rocm_platform",
    actual = "//xla/stream_executor/rocm:all_runtime",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        "manual",
    ]),
)

alias(
    name = "sycl_platform",
    actual = "//xla/stream_executor/sycl:all_runtime",
    tags = [
        "gpu",
    ],
)
