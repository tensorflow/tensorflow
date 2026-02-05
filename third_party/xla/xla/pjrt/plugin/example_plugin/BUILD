load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        "@tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "myplugin_cpp_pjrt_test",
    srcs = ["myplugin_cpp_pjrt_test.cc"],
    deps = [
        ":myplugin_cpp_pjrt",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
    ],
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
    name = "myplugin_c_pjrt_internal",
    srcs = [
        "myplugin_c_pjrt_internal.cc",
    ],
    hdrs = ["myplugin_c_pjrt_internal.h"],
    deps = [
        ":example_extension_impl",
        ":myplugin_cpp_pjrt",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/extensions/example:example_extension",
        "//xla/pjrt/extensions/example:example_extension_cpp",
        "//xla/pjrt/extensions/example:example_extension_private",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "myplugin_c_pjrt",
    srcs = [
        "myplugin_c_pjrt.cc",
    ],
    hdrs = ["myplugin_c_pjrt.h"],
    deps = [
        ":myplugin_c_pjrt_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
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
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_myplugin_plugin.so",
    linkshared = True,
    deps = [
        ":myplugin_c_pjrt",
    ],
)
