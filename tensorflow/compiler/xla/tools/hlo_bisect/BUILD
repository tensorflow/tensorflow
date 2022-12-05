# Description:
#   A tool for reducing a HLO module that produces incorrect results.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")

package(
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "hlo_bisect",
    testonly = True,
    srcs = ["hlo_bisect.cc"],
    deps = [
        ":hlo_bisect_utils",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:interpreter_plugin",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "hlo_bisect_state",
    srcs = ["hlo_bisect_state.cc"],
    hdrs = ["hlo_bisect_state.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "hlo_bisect_state_test",
    srcs = ["hlo_bisect_state_test.cc"],
    deps = [
        ":hlo_bisect_state",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "hlo_bisect_utils",
    testonly = True,
    srcs = ["hlo_bisect_utils.cc"],
    hdrs = ["hlo_bisect_utils.h"],
    deps = [
        ":hlo_bisect_state",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service:hlo_runner_interface",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tools:prepare_reference_module",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:subprocess",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
