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
        ":plugin_test_fixture",
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
    backends = ["nvgpu_any"],
    tags = [
        "gpu",
        "no_oss",
    ] + if_google(["config-cuda-only"]),
    deps = [
        ":plugin_test_fixture",
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

cc_library(
    name = "plugin_test_fixture",
    testonly = True,
    srcs = [
        "plugin_test_fixture.cc",
    ],
    hdrs = ["plugin_test_fixture.h"],
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
    ],
)

xla_cc_test(
    name = "plugin_registration_test_cpu",
    srcs = ["plugin_registration_test.cc"],
    deps = [
        ":plugin_test_fixture",
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

xla_cc_test(
    name = "plugin_compile_only_test",
    srcs = ["plugin_compile_only_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":plugin_test_fixture",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_tpu:tpu_static_registration",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)
