load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/mkl:graph.bzl",
    "onednn_graph_cc_library",
    "onednn_graph_cc_test",
)

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

onednn_graph_cc_library(
    name = "onednn_interop",
    hdrs = ["onednn_interop.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@onednn//:mkl_dnn",
    ],
)

onednn_graph_cc_library(
    name = "onednn_fusion_thunk",
    srcs = ["onednn_fusion_thunk.cc"],
    hdrs = ["onednn_fusion_thunk.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":onednn_interop",
        ":onednn_threadpool",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:onednn_fusion",
        "//xla/backends/cpu/runtime:dot_lib",
        "//xla/backends/cpu/runtime:parallel_loop_runner",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/runtime:buffer_use",
        "//xla/runtime:object_pool",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@onednn//:mkl_dnn",
    ],
)

onednn_graph_cc_library(
    name = "onednn_threadpool",
    hdrs = ["onednn_threadpool.h"],
    deps = [
        ":onednn_interop",
        "//xla/backends/cpu/runtime:work_queue",
        "@eigen_archive//:eigen3",
        "@onednn//:mkl_dnn",
    ],
)

onednn_graph_cc_test(
    name = "onednn_threadpool_test",
    srcs = ["onednn_threadpool_test.cc"],
    deps = [
        ":onednn_interop",
        ":onednn_threadpool",
        "//xla/backends/cpu/runtime:work_queue",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@onednn//:mkl_dnn",
    ],
)
