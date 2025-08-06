load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "if_nnapi")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
        "NeuralNetworksTypes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    linkopts = if_nnapi(["-ldl"]),
)

cc_library(
    name = "nnapi_implementation_headers",
    hdrs = ["nnapi_implementation.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_lib",
    ],
)

cc_library(
    name = "nnapi_implementation",
    srcs = if_nnapi(
        not_supported = ["nnapi_implementation_disabled.cc"],
        supported = ["nnapi_implementation.cc"],
    ),
    hdrs = ["nnapi_implementation.h"],
    compatible_with = get_compatible_with_portable(),
    linkopts = if_nnapi(["-ldl"]) + if_nnapi(
        supported = ["-lrt"],
        supported_android = [],
    ),
    deps = [
        ":nnapi_lib",
    ] + if_nnapi(["//tensorflow/lite/nnapi/sl:nnapi_support_library_headers"]),
)

# This target exists only to verify that nnapi_implementation_disabled.cc compiles.
cc_library(
    name = "nnapi_implementation_disabled",
    testonly = 1,
    srcs = ["nnapi_implementation_disabled.cc"],
    hdrs = ["nnapi_implementation.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [":nnapi_lib"],
)

cc_library(
    name = "nnapi_util",
    srcs = ["nnapi_util.cc"],
    hdrs = ["nnapi_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_implementation_headers",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "nnapi_implementation_test",
    srcs = ["nnapi_implementation_test.cc"],
    deps = [
        ":nnapi_implementation",
        "@com_google_googletest//:gtest_main",
    ],
)

# Cannot inject NNAPI instance on ios and windows
cc_library(
    name = "nnapi_handler",
    srcs = if_nnapi(["nnapi_handler.cc"]),
    hdrs = if_nnapi(["nnapi_handler.h"]),
    deps = [
        ":nnapi_implementation",
        ":nnapi_lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite:framework",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "nnapi_handler_test",
    srcs = ["nnapi_handler_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_handler",
        ":nnapi_implementation",
        "@com_google_googletest//:gtest_main",
    ],
)
