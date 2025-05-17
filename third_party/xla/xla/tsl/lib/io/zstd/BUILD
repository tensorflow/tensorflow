load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "zstd_output_buffer",
    srcs = ["zstd_output_buffer.cc"],
    hdrs = ["zstd_output_buffer.h"],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@net_zstd//:zstdlib",
    ],
)

tsl_cc_test(
    name = "zstd_output_buffer_test",
    srcs = ["zstd_output_buffer_test.cc"],
    deps = [
        ":zstd_output_buffer",
        "//xla/tsl/platform:mock_writable_file",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "zstd_input_stream",
    srcs = ["zstd_input_stream.cc"],
    hdrs = ["zstd_input_stream.h"],
    deps = [
        "//xla/tsl/lib/io:inputstream_interface",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:tstring",
        "@net_zstd//:zstdlib",
    ],
)

tsl_cc_test(
    name = "zstd_input_stream_test",
    srcs = ["zstd_input_stream_test.cc"],
    deps = [
        ":zstd_input_stream",
        "//xla/tsl/lib/io:mock_input_stream_interface",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:tstring",
    ],
)

tsl_cc_test(
    name = "zstd_test",
    srcs = ["zstd_test.cc"],
    deps = [
        ":zstd_input_stream",
        ":zstd_output_buffer",
        "//xla/tsl/lib/io:mock_input_stream_interface",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:mock_writable_file",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:tstring",
    ],
)
