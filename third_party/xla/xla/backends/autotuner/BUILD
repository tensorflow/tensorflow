load("//xla/tests:build_defs.bzl", "xla_test")

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
    name = "codegen_backend",
    hdrs = ["codegen_backend.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "autotuner",
    srcs = ["autotuner.cc"],
    hdrs = ["autotuner.h"],
    deps = [
        ":codegen_backend",
        ":executor",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

xla_test(
    name = "autotuner_test",
    srcs = ["autotuner_test.cc"],
    deps = [
        ":autotuner",
        ":codegen_backend",
        ":executor",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "executor",
    hdrs = ["executor.h"],
    deps = [
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)
