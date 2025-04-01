load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_gpu_pjrt_client",
    srcs = [
        "xla_gpu_pjrt_client.cc",
    ],
    hdrs = ["xla_gpu_pjrt_client.h"],
    deps = [
        ":xla_gpu_client_options",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt/gpu/tfrt:tfrt_gpu_client",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "xla_gpu_client_options",
    srcs = [],
    hdrs = ["xla_gpu_client_options.h"],
    deps = [
        ":xla_gpu_allocator_config",
        "//xla/pjrt/distributed:key_value_store_interface",
    ],
)

cc_library(
    name = "xla_gpu_allocator_config",
    srcs = [],
    hdrs = ["xla_gpu_allocator_config.h"],
    deps = [
    ],
)

xla_test(
    name = "xla_gpu_pjrt_client_test",
    srcs = ["xla_gpu_pjrt_client_test.cc"],
    backends = ["gpu_any"],
    tags = [
        "gpu",
        "no_oss",
    ] + if_google(["config-cuda-only"]),
    deps = [
        ":xla_gpu_client_options",
        ":xla_gpu_pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt/gpu/tfrt:tfrt_gpu_client",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)
