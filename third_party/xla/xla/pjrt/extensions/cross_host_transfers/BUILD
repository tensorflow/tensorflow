load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_cross_host_transfer_extension",
    srcs = ["pjrt_c_api_cross_host_transfers_extension.cc"],
    hdrs = ["pjrt_c_api_cross_host_transfers_extension.h"],
    deps = [
        "//xla:future",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "pjrt_c_api_cross_host_transfers_extension_gpu_test",
    srcs = ["pjrt_c_api_cross_host_transfers_extension_gpu_test.cc"],
    backend_tags = {
        "h100": [
            "full",
            "multi_gpu",
            "no_oss",
        ],
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    tags = if_google([
        "config-cuda-only",
    ]),
    deps = [
        ":pjrt_c_api_cross_host_transfer_extension",
        "//xla:debug_options_flags",
        "//xla:future",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/client:client_library",
        "//xla/ffi/api:ffi",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt/c:pjrt_c_api_gpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:service",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)
