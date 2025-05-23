load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "cpu_codegen_backend",
    hdrs = ["cpu_codegen_backend.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/host:host_platform",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,  # needed for platform registration
)

cc_library(
    name = "xnnpack_backend",
    srcs = ["xnnpack_backend.cc"],
    hdrs = ["xnnpack_backend.h"],
    deps = [
        ":cpu_codegen_backend",
        "//xla:status_macros",
        "//xla:util",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/backends/cpu:xnn_fusion",
        "//xla/backends/cpu:xnnpack_config_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "xnnpack_backend_test",
    srcs = ["xnnpack_backend_test.cc"],
    deps = [
        ":cpu_codegen_backend",
        ":xnnpack_backend",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:compiler",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:casts",
    ],
)
