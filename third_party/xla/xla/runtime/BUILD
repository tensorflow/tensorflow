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
    name = "thunk",
    hdrs = ["thunk.h"],
)
