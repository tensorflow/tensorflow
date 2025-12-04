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
    name = "ffi",
    hdrs = ["ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/ffi:api",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "@com_google_absl//absl/base:core_headers",
    ],
)
