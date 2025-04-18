load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "xnn_interop",
    srcs = ["xnn_interop.cc"],
    hdrs = ["xnn_interop.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
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
        "//xla/backends/cpu/runtime:parallel_loop_runner",
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
        ":xnn_threadpool",
        "//xla/backends/cpu/runtime:parallel_loop_runner",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@pthreadpool",
    ],
)

cc_library(
    name = "xnn_convolution_thunk",
    srcs = ["xnn_convolution_thunk.cc"],
    hdrs = ["xnn_convolution_thunk.h"],
    deps = [
        ":xnn_fusion_thunk",
        ":xnn_interop",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:convolution_lib",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "xnn_convolution_thunk_test",
    srcs = ["xnn_convolution_thunk_test.cc"],
    deps = [
        ":xnn_convolution_thunk",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
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
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_lib",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "xnn_dot_thunk_test",
    srcs = ["xnn_dot_thunk_test.cc"],
    deps = [
        ":xnn_dot_thunk",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "xnn_fusion_thunk",
    srcs = ["xnn_fusion_thunk.cc"],
    hdrs = ["xnn_fusion_thunk.h"],
    deps = [
        ":xnn_interop",
        ":xnn_threadpool",
        "//xla:shape_util",
        "//xla/backends/cpu/runtime:object_pool",
        "//xla/backends/cpu/runtime:parallel_loop_runner",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pthreadpool",
    ],
)

xla_cc_test(
    name = "xnn_fusion_thunk_test",
    srcs = ["xnn_fusion_thunk_test.cc"],
    deps = [
        ":xnn_fusion_thunk",
        ":xnn_interop",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@XNNPACK",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)
