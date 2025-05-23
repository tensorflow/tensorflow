load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

xla_cc_test(
    name = "plugin_registration_test_tpu",
    srcs = ["plugin_registration_test.cc"],
    tags = ["no_oss"],
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_tpu:tpu_static_registration",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "plugin_registration_test_gpu",
    srcs = ["plugin_registration_test.cc"],
    backends = ["gpu_any"],
    tags = [
        "gpu",
        "no_oss",
    ] + if_google(["config-cuda-only"]),
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "plugin_registration_test_cpu",
    srcs = ["plugin_registration_test.cc"],
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
