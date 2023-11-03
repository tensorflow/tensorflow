load("//xla:xla.bzl", "xla_cc_test")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "api",
    hdrs = ["//xla/ffi/api:api_headers"],
    visibility = ["//visibility:public"],
    deps = ["//xla/ffi/api:c_api"],
)

cc_library(
    name = "call_frame",
    srcs = ["call_frame.cc"],
    hdrs = ["call_frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ffi",
    srcs = ["ffi.cc"],
    hdrs = ["ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":call_frame",
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/runtime:memref_view",
        "//xla/service:executable",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    deps = [
        ":api",
        ":call_frame",
        ":ffi",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/service:executable",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
