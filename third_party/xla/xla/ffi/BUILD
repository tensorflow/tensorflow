load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "api",
    hdrs = ["//xla/ffi/api:api_headers"],
    visibility = ["//visibility:private"],
    deps = ["//xla/ffi/api:c_api"],
)

cc_library(
    name = "call_frame",
    srcs = ["call_frame.cc"],
    hdrs = ["call_frame.h"],
    deps = [
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        "//xla:shape_util",
        "//xla:status",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:memref_view",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ffi_api",
    srcs = ["ffi_api.cc"],
    hdrs = ["ffi_api.h"],
    deps = [
        ":api",
        ":call_frame",
        "//xla:status",
        "//xla:statusor",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    deps = [
        ":call_frame",
        ":ffi",
        ":ffi_api",
        "//xla:xla_data_proto_cc",
        "//xla/service:executable",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
