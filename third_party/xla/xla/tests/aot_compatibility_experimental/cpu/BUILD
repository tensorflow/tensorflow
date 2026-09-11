load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

xla_test(
    name = "elementwise_aot_test",
    srcs = ["elementwise_aot_test.cc"],
    backends = ["cpu"],
    data = glob(["executables/elementwise_aot_test_cpu/**/*.pbtxt"]),
    tags = [
        "manual",
        "no_oss",
    ],
    deps = [
        "//xla:literal",
        "//xla/service:hlo_module_config",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/aot_compatibility_experimental:test_lib",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
