load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)

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
        ":profiler",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "autotuner_test",
    srcs = ["autotuner_test.cc"],
    deps = [
        ":autotuner",
        ":codegen_backend",
        ":profiler",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:executable",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "profiler",
    hdrs = ["profiler.h"],
    deps = [
        "//xla/service:executable",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)
