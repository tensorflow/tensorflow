load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "buffer_allocations",
    srcs = ["buffer_allocations.cc"],
    hdrs = ["buffer_allocations.h"],
    deps = [
        "//xla/service:buffer_assignment",
        "//xla/service:maybe_owning_device_memory",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "thunk",
    srcs = ["thunk.cc"],
    hdrs = ["thunk.h"],
    deps = [
        ":buffer_allocations",
        "//xla/stream_executor/host:host_kernel_c_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_library(
    name = "call_thunk",
    srcs = ["call_thunk.cc"],
    hdrs = ["call_thunk.h"],
    deps = [
        ":thunk",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "copy_thunk",
    srcs = ["copy_thunk.cc"],
    hdrs = ["copy_thunk.h"],
    deps = [
        ":thunk",
        "//xla:shape_util",
        "//xla/pjrt:transpose",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "copy_thunk_test",
    srcs = ["copy_thunk_test.cc"],
    deps = [
        ":buffer_allocations",
        ":copy_thunk",
        ":thunk",
        "//xla:shape_util",
        "//xla/service:buffer_assignment",
        "//xla/service:maybe_owning_device_memory",
        "//xla/stream_executor",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "kernel_thunk",
    srcs = ["kernel_thunk.cc"],
    hdrs = ["kernel_thunk.h"],
    deps = [
        ":thunk",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_kernel",
        "//xla/stream_executor/host:host_kernel_c_api",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "kernel_thunk_test",
    srcs = ["kernel_thunk_test.cc"],
    deps = [
        ":buffer_allocations",
        ":kernel_thunk",
        ":thunk",
        "//xla/service:buffer_assignment",
        "//xla/service:maybe_owning_device_memory",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_kernel_c_api",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "while_thunk",
    srcs = ["while_thunk.cc"],
    hdrs = ["while_thunk.h"],
    deps = [
        ":thunk",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)
