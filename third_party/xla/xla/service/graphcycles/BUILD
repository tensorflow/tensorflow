load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("@local_tsl//tsl:tsl.bzl", "set_external_visibility")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ordered_set",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ordered_set",
    hdrs = ["ordered_set.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "graphcycles_test",
    srcs = ["graphcycles_test.cc"],
    deps = [
        ":graphcycles",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "ordered_set_test",
    srcs = ["ordered_set_test.cc"],
    deps = [
        ":ordered_set",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)
