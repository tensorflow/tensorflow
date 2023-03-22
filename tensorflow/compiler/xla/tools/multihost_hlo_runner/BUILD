load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow/tsl:tsl.bzl", "if_cuda_or_rocm")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

build_test(
    name = "hlo_runner_main_build_test",
    tags = [
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
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util:command_line_flags",
    ] + if_cuda_or_rocm([
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ]) + if_cuda([
        "//tensorflow/compiler/xla/stream_executor:cuda_platform",
    ]),
)

cc_library(
    name = "functional_hlo_runner",
    srcs = ["functional_hlo_runner.cc"],
    hdrs = ["functional_hlo_runner.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tools:hlo_control_flow_flattening",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_runner_flags",
    srcs = ["hlo_runner_flags.cc"],
    hdrs = ["hlo_runner_flags.h"],
    deps = [
        ":functional_hlo_runner",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)
