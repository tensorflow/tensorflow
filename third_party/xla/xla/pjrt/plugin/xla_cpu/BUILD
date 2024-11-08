load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_cpu_pjrt_client",
    srcs = [
        "xla_cpu_pjrt_client.cc",
    ],
    hdrs = ["xla_cpu_pjrt_client.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "xla_cpu_pjrt_client_test",
    srcs = ["xla_cpu_pjrt_client_test.cc"],
    deps = [
        ":xla_cpu_pjrt_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)
