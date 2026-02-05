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
    name = "slinky_threadpool",
    srcs = ["slinky_threadpool.cc"],
    hdrs = ["slinky_threadpool.h"],
    deps = [
        "//xla/backends/cpu/runtime:work_queue",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@slinky//slinky/base:thread_pool",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "slinky_threadpool_test",
    srcs = ["slinky_threadpool_test.cc"],
    deps = [
        ":slinky_threadpool",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings:str_format",
        "@slinky//slinky/base:thread_pool",
    ],
)

cc_library(
    name = "ynn_interop",
    srcs = ["ynn_interop.cc"],
    hdrs = ["ynn_interop.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "@XNNPACK//ynnpack",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ynn_threadpool",
    srcs = ["ynn_threadpool.cc"],
    hdrs = ["ynn_threadpool.h"],
    deps = [
        ":slinky_threadpool",
        ":ynn_interop",
        "@XNNPACK//ynnpack:ynnpack_h",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "ynn_fusion_thunk",
    srcs = ["ynn_fusion_thunk.cc"],
    hdrs = ["ynn_fusion_thunk.h"],
    deps = [
        ":ynn_interop",
        "//xla:shape_util",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:buffer_use",
        "//xla/runtime:object_pool",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_address",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK//ynnpack:ynnpack_h",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "ynn_fusion_thunk_test",
    srcs = ["ynn_fusion_thunk_test.cc"],
    deps = [
        ":ynn_fusion_thunk",
        ":ynn_interop",
        ":ynn_threadpool",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_testlib",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@XNNPACK//ynnpack:ynnpack_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)
