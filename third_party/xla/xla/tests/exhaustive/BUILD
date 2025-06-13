# Description:
#   Computationally expensive, exhaustive tests for XLA

load("//xla/tests:build_defs.bzl", "GPU_BACKENDS", "xla_test")
load("//xla/tests/exhaustive:build_defs.bzl", "exhaustive_xla_test")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:friends"],
    licenses = ["notice"],
)

filegroup(
    name = "exhaustive_op_test_utils_shared_hdrs",
    testonly = True,
    srcs = [
        "error_spec.h",
        "exhaustive_op_test_base.h",
        "exhaustive_op_test_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "exhaustive_op_test_utils_shared_srcs",
    testonly = True,
    srcs = [
        "exhaustive_op_test_base.cc",
        "exhaustive_op_test_utils.cc",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "exhaustive_op_test_utils",
    testonly = True,
    srcs = [
        "platform.cc",
        ":exhaustive_op_test_utils_shared_srcs",
    ],
    hdrs = [
        "exhaustive_op_test.h",
        "platform.h",
        "test_op.h",
        ":exhaustive_op_test_utils_shared_hdrs",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//xla:bit_cast",
        "//xla:executable_run_options",
        "//xla:fp_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/tests:client_library_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "exhaustive_unary_test_textual_hdrs",
    textual_hdrs = [
        "exhaustive_unary_test_definitions.inc",
        "exhaustive_unary_test_f32_and_smaller_instantiation.inc",
        "exhaustive_unary_test_f64_instantiation.inc",
        "exhaustive_unary_test_ops.inc",
    ],
)

cc_library(
    name = "exhaustive_test_main",
    testonly = True,
    srcs = ["exhaustive_test_main.cc"],
    deps = [
        ":exhaustive_op_test_utils",
        "//xla/tsl/util:command_line_flags",
        "@com_google_googletest//:gtest_for_library",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

exhaustive_xla_test(
    name = "exhaustive_unary_test",
    timeout = "long",
    srcs = [
        "exhaustive_unary_test_definitions.h",
        "exhaustive_unary_test_functions.cc",
    ],
    # Nvidia close-sourced libraries are not TSAN friendly, but are doing their
    # own synchronization. This can lead to TSAN false positives that are hard
    # to track down.
    backend_tags = {
        gpu: ["notsan"]
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    },
    backends = ["cpu"] + [
        gpu
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    ],
    # Partition goal is to get under ~5 minute execution time without needlessly
    # reserving hardware for each shard.
    #
    # exhaustive_xla_test needs to have all partition names added to allow other
    # build tools to function.
    partitions = {
        "f32_and_smaller": [
            "exhaustive_unary_test_f32_and_smaller_instantiation.cc",
        ],
        "f64": [
            ":exhaustive_unary_test_f64_instantiation.cc",
        ],
    },
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_no_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        ":exhaustive_test_main",
        ":exhaustive_unary_test_textual_hdrs",
        "//xla:literal",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:math",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "exhaustive_unary_complex_test",
    timeout = "long",
    srcs = [
        "exhaustive_unary_complex_test.cc",
    ],
    # Nvidia close-sourced libraries are not TSAN friendly, but are doing their
    # own synchronization. This can lead to TSAN false positives that are hard
    # to track down.
    backend_tags = {
        gpu: ["notsan"]
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    },
    backends = ["cpu"] + [
        gpu
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    ],
    shard_count = 50,
    tags = [
        "optonly",
        "test_xla_cpu_no_thunks",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_op_test_utils",
        ":exhaustive_test_main",
        "//xla:literal",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "exhaustive_binary_test_textual_hdrs",
    textual_hdrs = [
        "exhaustive_binary_test_definitions.inc",
        "exhaustive_binary_test_f16_and_smaller_instantiation.inc",
        "exhaustive_binary_test_f32_instantiation.inc",
        "exhaustive_binary_test_f64_instantiation.inc",
        "exhaustive_binary_test_ops.inc",
    ],
)

exhaustive_xla_test(
    name = "exhaustive_binary_test",
    timeout = "long",
    srcs = [
        "exhaustive_binary_test_definitions.h",
        "exhaustive_binary_test_functions.cc",
    ],
    # Nvidia close-sourced libraries are not TSAN friendly, but are doing their
    # own synchronization. This can lead to TSAN false positives that are hard
    # to track down.
    backend_tags = {
        gpu: ["notsan"]
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    },
    backends = ["cpu"] + [
        gpu
        for gpu in GPU_BACKENDS
        if gpu != "nvgpu_any" and gpu != "amdgpu_any"
    ],
    # Partition goal is to get under ~5 minute execution time without needlessly
    # reserving hardware for each shard.
    #
    # exhasutive_xla_test needs to have all partition names added to allow other
    # build tools to function.
    partitions = {
        "f16_and_smaller": [
            "exhaustive_binary_test_f16_and_smaller_instantiation.cc",
        ],
        "f32": [
            "exhaustive_binary_test_f32_instantiation.cc",
        ],
        "f64": [
            "exhaustive_binary_test_f64_instantiation.cc",
        ],
    },
    shard_count = 50,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
    ],
    deps = [
        ":exhaustive_binary_test_textual_hdrs",
        ":exhaustive_op_test_utils",
        ":exhaustive_test_main",
        "//xla:literal",
        "//xla:types",
        "//xla/hlo/builder:xla_builder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)
