load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "testing_cpp_pjrt",
    srcs = [
        "testing_cpp_pjrt.cc",
    ],
    hdrs = ["testing_cpp_pjrt.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "testing_cpp_pjrt_test",
    srcs = ["testing_cpp_pjrt_test.cc"],
    deps = [
        ":testing_cpp_pjrt",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "testing_c_pjrt_internal",
    srcs = [
        "testing_c_pjrt_internal.cc",
    ],
    hdrs = ["testing_c_pjrt_internal.h"],
    deps = [
        ":testing_cpp_pjrt",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "testing_c_pjrt_internal_test",
    srcs = ["testing_c_pjrt_internal_test.cc"],
    deps = [
        ":testing_c_pjrt_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
