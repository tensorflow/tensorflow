load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

cc_binary(
    name = "hlo_runner_main",
    srcs = ["hlo_runner_main.cc"],
    deps = [
        ":hlo_runner_flags",
        ":functional_hlo_runner",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:debug_options_flags",
        # This is needed to run on GPU.
        "//tensorflow/compiler/jit:xla_gpu_jit",
        # "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util:command_line_flags",
    ],
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
        "data/sharded_2_devices.hlo",
        "data/single_device.hlo",
    ],
    deps = [
        ":functional_hlo_runner",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)
