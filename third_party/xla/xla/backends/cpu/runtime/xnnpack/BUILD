load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "object_pool",
    hdrs = ["object_pool.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "object_pool_test",
    srcs = ["object_pool_test.cc"],
    deps = [
        ":object_pool",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "parallel_loop_runner",
    srcs = ["parallel_loop_runner.cc"],
    hdrs = ["parallel_loop_runner.h"],
    deps = [
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
    ],
)

xla_cc_test(
    name = "parallel_loop_runner_test",
    srcs = ["parallel_loop_runner_test.cc"],
    deps = [
        ":parallel_loop_runner",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "xnn_interop",
    srcs = ["xnn_interop.cc"],
    hdrs = ["xnn_interop.h"],
    deps = [
        "//xla:util",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "xnn_threadpool",
    srcs = ["xnn_threadpool.cc"],
    hdrs = ["xnn_threadpool.h"],
    # copybara:uncomment_begin(google-only)
    # local_defines = select({
    # "@pthreadpool:pthreadpool_header_only_explicit_true": [
    # "XLA_CPU_USE_CUSTOM_PTHREADPOOL",
    # ],
    # "//conditions:default": [],
    # }),
    # copybara:uncomment_end
    deps = [
        ":parallel_loop_runner",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:platform_port",
        "@pthreadpool",
    ],
)

xla_cc_test(
    name = "xnn_threadpool_test",
    srcs = ["xnn_threadpool_test.cc"],
    deps = [
        ":parallel_loop_runner",
        ":xnn_threadpool",
        "//xla/tsl/concurrency:async_value",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
        "@pthreadpool",
    ],
)

cc_library(
    name = "xnn_dot_thunk",
    srcs = ["xnn_dot_thunk.cc"],
    hdrs = ["xnn_dot_thunk.h"],
    deps = [
        ":xnn_fusion_thunk",
        ":xnn_interop",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_lib",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/service:buffer_assignment",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "xnn_dot_thunk_test",
    srcs = ["xnn_dot_thunk_test.cc"],
    deps = [
        ":xnn_dot_thunk",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "xnn_fusion_thunk",
    srcs = ["xnn_fusion_thunk.cc"],
    hdrs = ["xnn_fusion_thunk.h"],
    deps = [
        ":object_pool",
        ":parallel_loop_runner",
        ":xnn_interop",
        ":xnn_threadpool",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_lib",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@pthreadpool",
    ],
)

xla_cc_test(
    name = "xnn_fusion_thunk_test",
    srcs = ["xnn_fusion_thunk_test.cc"],
    deps = [
        ":xnn_fusion_thunk",
        ":xnn_interop",
        "//xla:executable_run_options",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)
