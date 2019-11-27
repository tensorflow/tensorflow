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
        "//tensorflow/lite/nnapi:nnapi_lib",
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
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "@com_google_googletest//:gtest_main",
    ],
)
