load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/megascale:package_groups.bzl", "megascale_package_groups")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":internal"]),
    licenses = ["notice"],
)

megascale_package_groups()

cc_library(
    name = "megascale_c_api_client",
    srcs = ["megascale_c_api_client.cc"],
    hdrs = ["megascale_c_api_client.h"],
    deps = [
        ":megascale_types",
        "//xla/megascale:addresses_proto_cc",
        "//xla/megascale:dcn_topology_proto_cc",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_megascale_extension_hdrs",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/pjrt/plugin:plugin_names",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "megascale_types",
    srcs = ["megascale_types.cc"],
    hdrs = ["megascale_types.h"],
    visibility = ["//xla/pjrt/c_api_client:__subpackages__"],
    deps = [
        "//xla/megascale:addresses_proto_cc",
        "//xla/megascale:dcn_topology_proto_cc",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_megascale_extension_hdrs",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

xla_cc_test(
    name = "megascale_c_api_client_test",
    srcs = ["megascale_c_api_client_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":megascale_c_api_client",
        "//xla/megascale:addresses_proto_cc",
        "//xla/megascale:dcn_topology_proto_cc",
        "//xla/pjrt/plugin/xla_tpu:tpu_static_registration",
        "//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest",
    ],
)
