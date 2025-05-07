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
    name = "buffer_use",
    srcs = ["buffer_use.cc"],
    hdrs = ["buffer_use.h"],
    deps = [
        "//xla/service:buffer_assignment",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "buffer_use_test",
    srcs = ["buffer_use_test.cc"],
    deps = [
        ":buffer_use",
        "//xla/service:buffer_assignment",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "resource_use",
    srcs = ["resource_use.cc"],
    hdrs = ["resource_use.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "resource_use_test",
    srcs = ["resource_use_test.cc"],
    deps = [
        ":resource_use",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "execution_graph",
    srcs = ["execution_graph.cc"],
    hdrs = ["execution_graph.h"],
    deps = [
        ":buffer_use",
        ":resource_use",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "execution_graph_test",
    srcs = ["execution_graph_test.cc"],
    deps = [
        ":buffer_use",
        ":execution_graph",
        ":resource_use",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "object_pool",
    hdrs = ["object_pool.h"],
    deps = [
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "object_pool_test",
    srcs = ["object_pool_test.cc"],
    deps = [
        ":object_pool",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "thunk",
    hdrs = ["thunk.h"],
)
