load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

#===-------------------------------------------------------------------------------------------===//
# Public XLA FFI API
#===-------------------------------------------------------------------------------------------===//

# XLA FFI is a header only library that does not have any dependencies on XLA. The intent is that
# users that do want to register custom FFI handlers with XLA should copy these headers to their
# project, build a shared object with an XLA FFI handler implementation, and load it at run time.
#
# `api.h` and `ffi.h` headers provide a C++ library for decoding XLA FFI C API structs into a more
# user friendly C++ types. Shared objects defining XLA FFI handlers should be built with private
# symbol visibility to avoid potential ODR violations coming from template instantiations of
# different XLA FFI versions.
#
# `ffi.h` defines builtin decoding for canonical XLA types, but users can add their own decodings
# with template specializations.

# A user of the FFI interface will only need the three headers included in the `all_headers`
# filegroup: `api.h`, `c_api.h`, and `ffi.h`.
filegroup(
    name = "all_headers",
    srcs = [
        "api.h",
        "c_api.h",
        "ffi.h",
    ],
)

filegroup(
    name = "api_headers",
    srcs = ["api.h"],
)

filegroup(
    name = "c_api_headers",
    srcs = ["c_api.h"],
)

cc_library(
    name = "api",
    hdrs = [":api_headers"],
    visibility = ["//visibility:private"],
    deps = [":c_api"],
)

cc_library(
    name = "c_api",
    hdrs = ["c_api.h"],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    deps = [":c_api"],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        ":c_api",
    ],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    deps = [
        ":c_api",
        ":ffi",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi:call_frame",
        "//xla/ffi:execution_context",
        "//xla/ffi:execution_state",
        "//xla/ffi:ffi_api",
        "//xla/ffi:type_id_registry",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)
