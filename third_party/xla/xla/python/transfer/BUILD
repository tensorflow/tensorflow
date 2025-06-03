load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_oss",
    "internal_visibility",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/python:friends",
        "//third_party/py/jax:__subpackages__",
    ]),
    licenses = ["notice"],
)

tf_proto_library(
    name = "transfer_socket_proto",
    srcs = [
        "transfer_socket.proto",
    ],
)

cc_library(
    name = "streaming",
    srcs = ["streaming.cc"],
    hdrs = ["streaming.h"],
    deps = [
        ":transfer_socket_proto_cc",
        "//xla/pjrt:pjrt_future",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "streaming_test",
    srcs = ["streaming_test.cc"],
    deps = [
        ":streaming",
        ":transfer_socket_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "event_loop",
    srcs = ["event_loop.cc"],
    hdrs = ["event_loop.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "event_loop_test",
    srcs = ["event_loop_test.cc"],
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":event_loop",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "streaming_ifrt",
    srcs = ["streaming_ifrt.cc"],
    hdrs = ["streaming_ifrt.h"],
    deps = [
        ":streaming",
        ":transfer_socket_proto_cc",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:raw_buffer",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "test_pattern",
    testonly = True,
    srcs = ["test_pattern.cc"],
    hdrs = ["test_pattern.h"],
    deps = [
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "streaming_ifrt_test",
    srcs = ["streaming_ifrt_test.cc"],
    deps = [
        ":streaming",
        ":streaming_ifrt",
        ":test_pattern",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_compiler",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:pjrt_cpu_client_test_lib",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket_bulk_transport",
    srcs = ["socket_bulk_transport.cc"],
    hdrs = ["socket_bulk_transport.h"],
    deps = [
        ":event_loop",
        ":streaming",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "socket_bulk_transport_test",
    srcs = ["socket_bulk_transport_test.cc"],
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":event_loop",
        ":socket_bulk_transport",
        ":streaming",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket-server",
    srcs = ["socket-server.cc"],
    hdrs = ["socket-server.h"],
    deps = [
        ":event_loop",
        ":streaming",
        ":transfer_socket_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "socket-server_test",
    srcs = ["socket-server_test.cc"],
    tags = if_oss(["not_run:arm"]),
    deps = [
        ":event_loop",
        ":socket-server",
        ":streaming",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
