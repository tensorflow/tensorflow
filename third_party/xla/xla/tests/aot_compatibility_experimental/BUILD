load("@rules_cc//cc:defs.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tests:hlo_test_base",
        "//xla/tests:pjrt_client_registry",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "test_lib_test",
    srcs = ["test_lib_test.cc"],
    data = glob(["gpu/executables/test_dummy_test/**"]),
    deps = [
        ":test_lib",
        "//xla/tests:aot_interception_pjrt_client",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)
