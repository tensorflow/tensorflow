load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "host_allocator_extension",
    srcs = ["host_allocator_extension.cc"],
    hdrs = ["host_allocator_extension.h"],
    deps = ["//xla/pjrt/c:pjrt_c_api_hdrs"],
)

cc_library(
    name = "host_allocator_interface_impl",
    srcs = ["host_allocator_interface_impl.cc"],
    hdrs = ["host_allocator_interface_impl.h"],
    deps = [
        ":host_allocator_extension",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "host_allocator_interface_impl_test",
    srcs = ["host_allocator_interface_impl_test.cc"],
    deps = [
        ":host_allocator_extension",
        ":host_allocator_interface_impl",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_googletest//:gtest_main",
    ],
)
