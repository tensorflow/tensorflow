load("//xla/tests:build_defs.bzl", "xla_test")

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

filegroup(
    name = "thunk_coverage_hlo",
    srcs = ["thunk_coverage.hlo"],
)

xla_test(
    name = "gemv_test",
    srcs = ["gemv_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "ynn_fusion_test",
    srcs = ["ynn_fusion_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
