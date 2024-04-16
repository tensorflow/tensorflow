load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_binary")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_cuda_or_rocm")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

build_test(
    name = "hlo_runner_main_build_test",
    tags = [
        "cpu",
        "gpu",
    ],
    targets = [
        ":hlo_runner_main",
    ],
)

xla_cc_binary(
    name = "hlo_runner_main",
    testonly = True,
    srcs = ["hlo_runner_main.cc"],
    tags = [
        "gpu",
        "noasan",  # Exceeds linker limit.
        "nomac",
    ],
    deps = [
        ":functional_hlo_runner",
        ":hlo_runner_flags",
        "//xla:debug_options_flags",
        "//xla:status",
        "//xla:statusor",
        "//xla/pjrt:pjrt_client",
        "//xla/service:cpu_plugin",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_or_rocm([
        "//xla/service:gpu_plugin",
    ]) + if_cuda([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor:rocm_platform",
    ]),
)

cc_library(
    name = "functional_hlo_runner",
    srcs = ["functional_hlo_runner.cc"],
    hdrs = ["functional_hlo_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_control_flow_flattening",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_runner_flags",
    srcs = ["hlo_runner_flags.cc"],
    hdrs = ["hlo_runner_flags.h"],
    deps = [
        ":functional_hlo_runner",
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "functional_hlo_runner_test",
    srcs = ["functional_hlo_runner_test.cc"],
    backend_tags = {
        # This test is tagged "manual" because it requires multiple (2) GPUs.
        "gpu": [
            "manual",
            "multi_gpu",
            "no_oss",
            "notap",
        ],
    },
    backends = ["gpu"],
    data = [
        "data/sharded_16_devices.hlo",
        "data/sharded_2_devices.hlo",
        "data/single_device.hlo",
        "data/single_device_tupled.hlo",
    ],
    tags = ["nomac"],
    deps = [
        ":functional_hlo_runner",
        "//xla/tests:filecheck",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)
