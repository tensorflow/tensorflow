load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "generate_benchmark_matrices_cc",
    srcs = ["generate_benchmark_matrices.cc"],
    hdrs = ["generate_benchmark_matrices.h"],
    deps = [
        "//xla/tools/benchmarks/proto:benchmark_config_proto_cc",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@jsoncpp_git//:jsoncpp",  # buildcleaner: keep
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "generate_benchmark_matrices_test",
    srcs = ["generate_benchmark_matrices_test.cc"],
    deps = [
        ":generate_benchmark_matrices_cc",
        "//xla/tools/benchmarks/proto:benchmark_config_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:path",
    ],
)

xla_cc_binary(
    name = "generate_benchmark_matrices_main",
    srcs = ["generate_benchmark_matrices_main.cc"],
    deps = [
        ":generate_benchmark_matrices_cc",
        "//xla/tools/benchmarks/proto:benchmark_config_proto_cc",
        "//xla/tsl/platform:status",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:platform_port",
    ],
)
