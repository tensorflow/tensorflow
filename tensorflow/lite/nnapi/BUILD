load("//tensorflow/lite:special_rules.bzl", "if_nnapi")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
        "NeuralNetworksTypes.h",
    ],
    linkopts = if_nnapi(["-ldl"]),
)

cc_library(
    name = "nnapi_implementation",
    srcs = if_nnapi(
        not_supported = ["nnapi_implementation_disabled.cc"],
        supported = ["nnapi_implementation.cc"],
    ),
    hdrs = [
        "nnapi_implementation.h",
    ],
    linkopts = if_nnapi(["-ldl"]) + if_nnapi(
        supported = ["-lrt"],
        supported_android = [],
    ),
    deps = [
        ":nnapi_lib",
    ],
)

cc_library(
    name = "nnapi_util",
    srcs = ["nnapi_util.cc"],
    hdrs = ["nnapi_util.h"],
    deps = [
        ":nnapi_implementation",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
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
