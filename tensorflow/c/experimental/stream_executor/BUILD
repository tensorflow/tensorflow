# Description:
# StreamExecutor C API.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "headers",
    srcs = [
        "stream_executor.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "stream_executor_hdrs",
    hdrs = ["stream_executor.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "stream_executor",
    srcs = ["stream_executor.cc"],
    hdrs = ["stream_executor.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":stream_executor_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/device:device_utils",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
        "@xla//xla/stream_executor:allocator_stats",
        "@xla//xla/stream_executor:device_description",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:executor_cache",
        "@xla//xla/stream_executor:generic_memory_allocation",
        "@xla//xla/stream_executor:generic_memory_allocator",
        "@xla//xla/stream_executor:memory_allocation",
        "@xla//xla/stream_executor:memory_allocator",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_common",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/tsl/framework:allocator",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "stream_executor_internal",
    hdrs = [
        "stream_executor.h",
        "stream_executor_internal.h",
    ],
    visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/core/common_runtime/pluggable_device:__subpackages__",
    ],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:executor_cache",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_common",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "stream_executor_test",
    srcs = ["stream_executor_test.cc"],
    deps = [
        ":stream_executor",
        ":stream_executor_internal",
        ":stream_executor_test_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "stream_executor_test_util",
    srcs = ["stream_executor_test_util.cc"],
    hdrs = ["stream_executor_test_util.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":stream_executor_hdrs",
        "//tensorflow/c:tf_status",
    ],
)
