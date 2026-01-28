load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "unstable_reduction_finder",
    srcs = ["unstable_reduction_finder.cc"],
    hdrs = ["unstable_reduction_finder.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
    ],
)

xla_cc_test(
    name = "unstable_reduction_finder_test",
    srcs = ["unstable_reduction_finder_test.cc"],
    deps = [
        ":unstable_reduction_finder",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "unstable_reduction_detector",
    srcs = ["unstable_reduction_detector.cc"],
    hdrs = ["unstable_reduction_detector.h"],
    deps = [
        ":unstable_reduction_finder",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "unstable_reduction_detector_test",
    srcs = ["unstable_reduction_detector_test.cc"],
    deps = [
        ":unstable_reduction_detector",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "float_check_device_test",
    srcs = ["float_check_device_test.cc"],
    backend_args = if_google(
        {
            "b200": ["--heap_check="],
            "h100": ["--heap_check="],
        },
        {},
    ),
    backends = [
        "h100",
        "b200",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
