licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
        "NeuralNetworksTypes.h",
    ],
    linkopts = ["-ldl"],
)

cc_library(
    name = "nnapi_implementation",
    srcs = select({
        "//tensorflow:ios": [
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
    linkopts = ["-ldl"] + select({
        "//tensorflow:android": [],
        "//tensorflow:macos": [],
        "//tensorflow:ios": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        "//tensorflow/lite/nnapi:nnapi_lib",
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
