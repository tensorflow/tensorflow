load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "async_value.h",
        "async_value_ref.h",
        "chain.h",
        "concurrent_vector.h",
        "executor.h",
        "ref_count.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "async_value.cc",
        "async_value_ref.cc",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "async_value",
    srcs = [
        "async_value.cc",
        "async_value_ref.cc",
    ],
    hdrs = [
        "async_value.h",
        "async_value_ref.h",
        "chain.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":concurrent_vector",
        ":executor",
        ":ref_count",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:context",
    ],
)

tsl_cc_test(
    name = "async_value_test",
    srcs = ["async_value_test.cc"],
    deps = [
        ":async_value",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
    ],
)

tsl_cc_test(
    name = "async_value_ptr_test",
    srcs = ["async_value_ptr_test.cc"],
    deps = [
        ":async_value",
        ":executor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

tsl_cc_test(
    name = "async_value_ref_test",
    srcs = ["async_value_ref_test.cc"],
    deps = [
        ":async_value",
        ":executor",
        ":ref_count",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "concurrent_vector",
    hdrs = ["concurrent_vector.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tsl_cc_test(
    name = "concurrent_vector_test",
    srcs = ["concurrent_vector_test.cc"],
    deps = [
        ":concurrent_vector",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",  # buildcleaner: keep
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "executor",
    hdrs = ["executor.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

cc_library(
    name = "future",
    srcs = ["future.cc"],
    hdrs = ["future.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_value",
        ":executor",
        ":ref_count",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tsl_cc_test(
    name = "future_test",
    srcs = ["future_test.cc"],
    deps = [
        ":executor",
        ":future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ref_count",
    hdrs = ["ref_count.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["@com_google_absl//absl/base:nullability"],
)

tsl_cc_test(
    name = "ref_count_test",
    srcs = ["ref_count_test.cc"],
    deps = [
        ":ref_count",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/base:nullability",
    ],
)
