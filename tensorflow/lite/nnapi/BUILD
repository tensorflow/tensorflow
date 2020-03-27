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
    linkopts = select({
        "//tensorflow:emscripten": [],
        "//tensorflow:ios": [],
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
)

cc_library(
    name = "nnapi_implementation",
    srcs = select({
        "//tensorflow:emscripten": [
            "nnapi_implementation_disabled.cc",
        ],
        "//tensorflow:ios": [
            "nnapi_implementation_disabled.cc",
        ],
        "//tensorflow:macos": [
            "nnapi_implementation_disabled.cc",
        ],
        "//tensorflow:windows": [
            "nnapi_implementation_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_implementation.cc",
        ],
    }),
    hdrs = [
        "nnapi_implementation.h",
    ],
    linkopts = select({
        "//tensorflow:emscripten": [],
        "//tensorflow:ios": [],
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:emscripten": [],
        "//tensorflow:ios": [],
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-lrt"],
    }),
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
    srcs = select({
        "//tensorflow:ios": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["nnapi_handler.cc"],
    }),
    hdrs = select({
        "//tensorflow:ios": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["nnapi_handler.h"],
    }),
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
