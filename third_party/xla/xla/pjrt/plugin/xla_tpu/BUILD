load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_tpu_pjrt_client",
    srcs = [
        "xla_tpu_pjrt_client.cc",
    ],
    hdrs = ["xla_tpu_pjrt_client.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "xla_tpu_pjrt_client_test",
    srcs = ["xla_tpu_pjrt_client_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":xla_tpu_pjrt_client",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
    ],
)
