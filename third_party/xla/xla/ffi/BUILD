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
        ":api",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "execution_context",
    srcs = ["execution_context.cc"],
    hdrs = ["execution_context.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/lib/gtl:int_type",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.cc"],
    deps = [
        ":execution_context",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        ":execution_context",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:scratch_allocator",
        "@com_google_absl//absl/status",
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
        ":execution_context",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    deps = [
        ":call_frame",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    deps = [
        ":call_frame",
        ":execution_context",
        ":ffi",
        ":ffi_api",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
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
