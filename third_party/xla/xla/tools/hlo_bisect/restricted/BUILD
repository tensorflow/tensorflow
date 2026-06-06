load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
)

xla_cc_binary(
    name = "hlo_bisect",
    testonly = True,
    srcs = ["hlo_bisect.cc"],
    deps = [
        ":hlo_bisect_utils",
        "//xla:debug_options_flags",
        "//xla/service:backend",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "//xla/service:interpreter_plugin",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda(["//xla/stream_executor/cuda:cublas_plugin"]),
)

build_test(
    name = "hlo_bisect_build_test",
    targets = [
        "//xla/tools/hlo_bisect/restricted:hlo_bisect",
    ],
)

cc_library(
    name = "hlo_bisect_utils",
    testonly = True,
    srcs = ["hlo_bisect_utils.cc"],
    hdrs = ["hlo_bisect_utils.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/service:platform_util",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:platform",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tools:prepare_reference_module",
        "//xla/tools/hlo_bisect:hlo_bisect_state",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform:path",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("hlo_bisect"),
    xla_restricted_allow("hlo_bisect_build_test"),
    xla_restricted_allow("hlo_bisect_build_test_0__deps"),
    xla_restricted_allow("hlo_bisect_utils"),
])
