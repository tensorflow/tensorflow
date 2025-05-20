load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
        ":tpu_static_registration",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt/c:pjrt_c_api_tpu_hdrs",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/plugin:plugin_names",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tpu_static_registration",
    srcs = ["tpu_static_registration.cc"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt/c:pjrt_c_api_tpu_hdrs",
        "//xla/pjrt/plugin:plugin_names",
        "//xla/pjrt/plugin:static_registration",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "xla_tpu_pjrt_client_test",
    srcs = ["xla_tpu_pjrt_client_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tpu_static_registration",
        ":xla_tpu_pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
    ],
)
