load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "myplugin_cpp_pjrt",
    srcs = [
        "myplugin_cpp_pjrt.cc",
    ],
    hdrs = ["myplugin_cpp_pjrt.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "myplugin_cpp_pjrt_test",
    srcs = ["myplugin_cpp_pjrt_test.cc"],
    deps = [
        ":myplugin_cpp_pjrt",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "myplugin_c_pjrt",
    srcs = [
        "myplugin_c_pjrt.cc",
    ],
    hdrs = ["myplugin_c_pjrt.h"],
    deps = [
        ":myplugin_cpp_pjrt",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "myplugin_c_pjrt_test",
    srcs = ["myplugin_c_pjrt_test.cc"],
    deps = [
        ":myplugin_c_pjrt",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_myplugin_plugin.so",
    linkshared = True,
    deps = [
        ":myplugin_c_pjrt",
    ],
)
