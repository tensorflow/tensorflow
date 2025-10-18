load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "example_extension_impl",
    srcs = ["example_extension_impl.cc"],
    hdrs = ["example_extension_impl.h"],
    deps = [
        "//xla/pjrt/extensions/example:example_extension_cpp",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "myplugin_c_pjrt",
    srcs = [
        "myplugin_c_pjrt.cc",
    ],
    hdrs = ["myplugin_c_pjrt.h"],
    deps = [
        ":example_extension_impl",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/extensions/example:example_extension",
        "//xla/pjrt/extensions/example:example_extension_cpp",
        "//xla/pjrt/extensions/example:example_extension_private",
        "//xla/pjrt/plugin/testing:testing_c_pjrt_internal",
    ],
)

xla_cc_test(
    name = "myplugin_c_pjrt_test",
    srcs = ["myplugin_c_pjrt_test.cc"],
    deps = [
        ":myplugin_c_pjrt",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/extensions/example:example_extension",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:path",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_myplugin_plugin.so",
    linkshared = True,
    deps = [
        ":myplugin_c_pjrt",
    ],
)
